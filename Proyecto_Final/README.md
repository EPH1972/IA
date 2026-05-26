# Aprendizaje por Refuerzo en StarCraft: Brood War mediante TorchCraft y PPO

**Autor:** Eduard Pérez  
**Fecha:** Mayo 2026  
**Estado:** Entrenamiento activo

---

## Resumen

Este trabajo presenta el diseño, implementación y puesta en marcha de un agente de aprendizaje por refuerzo (RL) capaz de jugar a *StarCraft: Brood War 1.16.1* en un escenario de micro-management de combate. El agente se ejecuta en un host Windows 10 y se comunica con el juego a través del protocolo TorchCraft ZMQ, que expone el estado estructurado del juego (posiciones, HP, tipos de unidad) mediante BWAPI 4.4.0 inyectado en el proceso de StarCraft dentro de una máquina virtual Windows 7. El algoritmo de aprendizaje es **Proximal Policy Optimization (PPO)** con una red MLP que consume un vector de observación de 189 características. El cliente TorchCraft es una reimplementación completa en Python puro con FlatBuffers manual, prescindiendo de la extensión C que ya no está disponible en PyPI.

El escenario de combate evolucionó durante el desarrollo: se comenzó con `m5v5_c_far.scm` (5 Marines Terran vs. 5 Zerglings) y posteriormente se migró a `dragoons_zealots.scm` (8 Dragoons/Zealots Protoss vs. 8 Dragoons/Zealots Protoss), un escenario de espejo simétrico que presenta desafíos adicionales de identificación de equipos al reportar BWEnv todos los `player_id` como 0.

---

## Índice

1. [Introducción](#1-introducción)
2. [Entorno de Ejecución](#2-entorno-de-ejecución)
3. [Pila de Software dentro de la VM](#3-pila-de-software-dentro-de-la-vm)
4. [Arquitectura del Sistema](#4-arquitectura-del-sistema)
5. [Cómo se Comunica el Modelo con BWAPI](#4b-cómo-se-comunica-el-modelo-con-bwapi)
6. [Protocolo TorchCraft](#5-protocolo-torchcraft)
7. [Espacio de Observación](#6-espacio-de-observación)
8. [Espacio de Acciones](#7-espacio-de-acciones)
9. [Algoritmo de Aprendizaje — PPO](#8-algoritmo-de-aprendizaje--ppo)
10. [Recompensa](#9-recompensa)
11. [Errores de Desarrollo y Soluciones](#10-errores-de-desarrollo-y-soluciones) (16 errores documentados)
12. [Uso Rápido](#11-uso-rápido)
13. [Recursos Utilizados](#12-recursos-utilizados)
14. [Trabajo Futuro](#13-trabajo-futuro)
15. [Referencias](#14-referencias)

---

## 1. Introducción

*StarCraft: Brood War* es uno de los entornos de referencia más complejos para agentes artificiales. Su espacio de acciones masivo, la información imperfecta y la necesidad de micro-gestión táctica en tiempo real lo convierten en un banco de pruebas ideal para algoritmos de RL modernos. Este proyecto se centra en el problema de **micro-management de combate**: controlar un grupo de unidades Protoss para derrotar a un grupo enemigo equivalente en el menor número de frames posible.

El escenario actual es el mapa `dragoons_zealots.scm` (5 unidades Protoss vs. 5 unidades Protoss (2 Dragoons + 3 Zealots), posiciones simétricas). Anteriormente se usó `m5v5_c_far.scm` (5 Marines Terran vs. 5 Zerglings), que sirvió para la fase inicial de integración del protocolo. El mapa de Protoss fue adoptado para estudiar comportamientos de micro-combate en un escenario de espejo simétrico, donde la identificación de equipos requiere inferencia posicional (ver Error 12).

### ¿Por qué TorchCraft y no píxeles?

Un enfoque basado en captura de pantalla + OCR fue considerado inicialmente. Se descartó por:

- **Latencia:** `VBoxManage screenshotpng` introduce ~150 ms por frame, incompatible con el bucle de juego.
- **Fragilidad:** OCR del HUD dependiente de resolución, fuentes y región exacta de captura.
- **Riqueza:** El estado estructurado de TorchCraft provee posiciones en píxeles, HP exacto, tipo de unidad y flags de estado sin ambigüedad ni ruido visual.

---

## 2. Entorno de Ejecución

### 2.1 Host

| Parámetro | Valor |
|---|---|
| Sistema operativo | Windows 10 Pro 22H2 |
| Python | 3.13 |
| CUDA | Disponible (entrenamiento en GPU) |
| Hypervisor | Oracle VirtualBox 7.x |
| Gestión VM | Vagrant |

### 2.2 Máquina Virtual

| Parámetro | Valor |
|---|---|
| Sistema operativo invitado | Windows 7 Ultimate SP1 x64 |
| RAM asignada | 8 GB |
| CPUs asignadas | 4 |
| Controlador gráfico | VBoxVGA |
| VRAM | 128 MB |
| NIC 1 | NAT (10.0.2.15) — acceso a internet |
| NIC 2 | Bridge (192.168.0.23) — comunicación con host |
| Resolución | 640 × 480 en ventana (400 × 300) |

La VM se gestiona con Vagrant. El host se conecta a BWEnv a través del **adaptador bridge** (`192.168.0.23:11111`), no mediante port-forwarding NAT, por razones que se detallan en la sección de errores.

### 2.3 Carpeta Compartida

Vagrant monta la carpeta del proyecto en `S:\` dentro de la VM. Esto permite editar archivos de configuración (`bwapi.ini`, `torchcraft.ini`) directamente desde el host en la ruta `VBox\Local Disk\Archivos de programa\Starcraft\bwapi-data\`, que se refleja en `C:\Archivos de programa\Starcraft\bwapi-data\` dentro de la VM.

---

## 3. Pila de Software dentro de la VM

La VM ejecuta una cadena de cuatro componentes que deben iniciarse en orden para que el sistema funcione. Cada uno tiene un rol específico e independiente.

```
StarCraft.exe  ←─── ChaosLauncher (inyector)
     │
     └── BWAPI 4.4.0 (hook dentro del proceso)
               │
               └── BWEnv.dll (plugin de IA de BWAPI)
                         │
                         └── ZMQ REP socket :11111
```

### 3.1 StarCraft: Brood War 1.16.1

El juego base. Se ejecuta en modo ventana (400×300 px) configurado en `torchcraft.ini`. La versión 1.16.1 es la última compatible con BWAPI 4.4.0; versiones posteriores (Remastered) usan una API diferente e incompatible.

**Instalación:**
1. StarCraft base 1.00 (desde imagen ISO española)
2. Expansión Brood War
3. Parche 1.16.1 (`BW-1161.exe`)
4. Wrapper DirectDraw `cnc-ddraw` para compatibilidad de color de 256 bits en VirtualBox

El wrapper `cnc-ddraw` es necesario porque VirtualBox no emula DirectDraw 256 colores de forma nativa; sin él, StarCraft arranca con la pantalla en negro o con artefactos de color.

### 3.2 ChaosLauncher

ChaosLauncher es un inyector de DLLs para StarCraft BW. Su función es cargar `BWAPI.dll` en el proceso de StarCraft antes de que empiece la ejecución del juego, habilitando el hook de BWAPI.

**Configuración relevante:**
- Plugin activo: `BWAPI Injector 4.4.0 [RELEASE]`
- NO se utiliza `bwheadless.exe` porque requiere modo LAN, que exige una partida multijugador. Para modo SINGLE_PLAYER con auto_menu se utiliza ChaosLauncher.

**Flujo de arranque:**
1. El usuario lanza `ChaosLauncher.exe`
2. ChaosLauncher inyecta `BWAPI.dll` en el proceso de StarCraft al iniciarlo
3. BWAPI lee `bwapi.ini` y carga el plugin de IA (`BWEnv.dll`)
4. `auto_menu` navega automáticamente por los menús hasta crear una partida

### 3.3 BWAPI 4.4.0

BWAPI (*Brood War Application Programming Interface*) es un hook a nivel de proceso que intercepta las llamadas internas de StarCraft y expone una API C++ para leer el estado del juego y enviar comandos. Se ejecuta dentro del proceso de StarCraft en el mismo espacio de memoria.

**Archivos relevantes:**
- `bwapi-data\BWAPI.dll` — la DLL principal del hook
- `bwapi-data\bwapi.ini` — configuración de auto_menu y plugin de IA
- `bwapi-data\AI\BWEnv.dll` — el plugin de IA cargado por BWAPI

**`bwapi.ini` configuración final:**

```ini
[ai]
ai     = bwapi-data/AI/BWEnv.dll
ai_dbg = bwapi-data/AI/BWEnv.dll

[auto_menu]
auto_menu    = SINGLE_PLAYER
auto_restart = ON
map          = Maps/BroodWar/micro/dragoons_zealots.scm
race         = Protoss
enemy_count  = 1
enemy_race   = Protoss
game_type    = USE_MAP_SETTINGS

[config]
shared_memory = ON

[window]
windowed = ON
width    = 400
height   = 300

[starcraft]
sound = OFF
```

La clave `game_type = USE_MAP_SETTINGS` es crítica: los mapas de micro-management tienen sus propias condiciones de inicio y no tienen *starting locations* para modo MELEE. Con `MELEE`, StarCraft no puede colocar las unidades y la partida se bloquea indefinidamente.

### 3.4 BWEnv.dll — Servidor TorchCraft

BWEnv.dll es el componente central. Es un plugin BWAPI que actúa como **servidor ZMQ REP** en el puerto 11111. Cada frame de juego, BWEnv:

1. Serializa el estado completo del juego (unidades, recursos, posiciones) en un mensaje FlatBuffers `StateUpdate`.
2. Espera a que el cliente Python envíe un mensaje `Commands`.
3. Ejecuta los comandos recibidos via la API BWAPI.
4. Avanza al siguiente frame y repite.

**`torchcraft.ini` configuración final:**

```ini
[general]
port = 11111
log_path = C:/tc_data/torchcraft_log_cpp_port_
display_log = false
img_mode = raw
window_mode = windows

[starcraft]
assume_on = true
launcher = bwheadless
```

La opción `assume_on = true` es crítica: indica a BWEnv que StarCraft ya está corriendo (lanzado por ChaosLauncher) y que no debe intentar lanzarlo él mismo. Con `assume_on = false`, BWEnv intenta arrancar StarCraft desde dentro de su propio proceso, causando un bloqueo inmediato.

**Protocolo de handshake:**
```
Cliente (Python)                    BWEnv.dll (VM)
     │                                    │
     │── HandshakeClient{protocol=30} ──► │
     │                                    │
     │◄── HandshakeServer{map, player_id, │
     │         map_w, map_h, lag} ────────│
     │                                    │
     │── Commands{[...]} ───────────────► │ (bucle de juego)
     │◄── StateUpdate{Frame|FrameDiff} ───│
     │           ...                      │
```

---

## 4. Arquitectura del Sistema

```
Host Windows 10
│
├── python main.py
│     │
│     ├── TorchCraftClient ──ZMQ REQ──► 192.168.0.23:11111
│     │     └── proto.py (FlatBuffers puro Python)
│     │           HandshakeClient{protocol=30} → HandshakeServer
│     │           Commands{[cmd, uid, x, y]}   → StateUpdate{GameState}
│     │
│     ├── SC1EnvTC (gymnasium.Env)
│     │     ├── StateEncoder  → vector float32 (189 features)
│     │     ├── CommandExecutor → lista de comandos BWAPI
│     │     ├── TCRewardCalculator → recompensa por combate
│     │     └── ActionLogger  → logs/action_log.jsonl
│     │
│     ├── PPOAgent
│     │     ├── ActorCriticMLP  (189 → 256 → 256 → actor/critic)
│     │     └── RolloutBuffer   (GAE λ=0.95)
│     │
│     └── Trainer
│           ├── Rollout collection (2048 pasos)
│           ├── PPO update (4 épocas, mini-batch 64)
│           └── Checkpoint cada 1000 pasos
│
│         ZMQ bridge 192.168.0.23:11111
└─────────────────────────────────────────────────
                  │
┌─────────────────▼───────────────────────────────┐
│          VM Windows 7 — StarCraft BW 1.16.1      │
│                                                  │
│  ChaosLauncher                                   │
│    └── StarCraft.exe (ventana 400×300)           │
│          └── BWAPI 4.4.0 (hook en proceso)       │
│                └── BWEnv.dll                     │
│                      └── ZMQ REP :11111          │
│                                                  │
│  Mapa: Maps/BroodWar/micro/dragoons_zealots.scm  │
│  2 Dragoons + 3 Zealots (jugador 0)  vs.  2 Dragoons + 3 Zealots (IA)      │
└──────────────────────────────────────────────────┘
```

---

## 4b. Cómo se Comunica el Modelo con BWAPI

Esta sección describe el flujo completo de datos desde la red neuronal PPO hasta los comandos que mueven las unidades dentro de StarCraft, y viceversa.

### Flujo de decisión (de red → juego)

```
ActorCriticMLP
│  recibe obs[189]
│  produce acción discreta ∈ {0..64}
▼
decode_tc_action(accion)          # action_space.py
│  0        → TCAction(NOOP)
│  1..64    → TCAction(ATTACK_MOVE, grid_row, grid_col)
▼
CommandExecutor.build_commands(action, state)   # command_executor.py
│  Convierte grid (row,col) a píxeles (x,y):
│    x = (col + 0.5) / 8 * map_w_px
│    y = (row + 0.5) / 8 * map_h_px
│  Filtra unidades del equipo propio (_own_unit_ids)
│  Para cada unidad propia genera:
│    [21, uid, 1, -1, x, y, 0]
│     ↑    ↑   ↑   ↑  ↑  ↑  └─ extra (0)
│     │    │   │   │  └──┘─── coordenadas en píxeles
│     │    │   │   └──── target_uid (-1 = posición, no unidad)
│     │    │   └──────── BWAPI UnitCommandType::Attack_Move = 1
│     │    └──────────── ID de unidad BWAPI
│     └───────────────── TC_CMD_UNIT_PROTECTED = 21
▼
TorchCraftClient.send(commands)   # client.py
│  Serializa con FlatBuffers:
│    Message { msg: Commands{[Command{code,args}]}, uid }
│  Envía por ZMQ REQ → 192.168.0.23:11111
▼
BWEnv.dll (VM)
│  Recibe Commands via ZMQ REP
│  Por cada comando con code=21 (COMMAND_UNIT_PROTECTED):
│    BWAPI::Broodwar->getUnit(uid)->attack(Position(x,y))
│    (solo ejecuta si la unidad pertenece al jugador 0)
▼
StarCraft.exe
   Mueve la unidad y auto-ataca enemigos en rango
```

### Flujo de observación (de juego → red)

```
StarCraft.exe
│  Avanza un frame (≈42 ms a 24 FPS)
▼
BWAPI 4.4.0 (hook en proceso)
│  Lee estado interno: posiciones, HP, flags, recursos
▼
BWEnv.dll
│  Serializa Frame o FrameDiff en FlatBuffers StateUpdate
│  Envía como respuesta ZMQ REP
▼
TorchCraftClient._process_state()  # client.py
│  Decodifica FlatBuffers → GameState
│  Si FrameDiff: aplica delta sobre prev_state.units
│  Si EndGame:   state.game_ended = True
▼
SC1EnvTC.step() / reset()
│  Limpia unidades stale (ver Error 16)
▼
StateEncoder.encode(state)         # state_encoder.py
│  Extrae y normaliza:
│    recursos[4] + workers[40] + army[60] + buildings[40]
│    + enemies[40] + summary[5]  →  obs[189]
│  Clasifica propios/enemigos por _own_unit_ids (split posicional)
▼
TCRewardCalculator.compute(state, action)  # reward.py
│  Compara con frame anterior:
│    kills = prev_enemy_count - enemy_count  → +0.50/baja
│    deaths = prev_army_count - army_count   → −0.50/baja
│    hp_lost = prev_enemy_hp - enemy_hp      → +0.001/HP
│    dist_delta = prev_avg_dist - avg_dist   → ±0.0005/px
│    survival                                → +0.001/frame
│    victoria (enemy_count == 0)             → +10.0
│    derrota  (army_count  == 0)             → −5.0
▼
ActorCriticMLP
   recibe obs[189] → genera siguiente acción
```

### Identificación de equipos: split posicional

Como BWEnv reporta `player_id=0` para **todas** las unidades (ver Error 12), el sistema no puede distinguir equipos por jugador. La solución implementada es un **split diagonal** basado en las posiciones iniciales:

```python
# En TCRewardCalculator, StateEncoder y CommandExecutor
all_u.sort(key=lambda u: u.x + u.y)   # diagonal x+y
mid = len(all_u) // 2
_own_unit_ids   = frozenset(u.id for u in all_u[:mid])   # esquina inferior
_enemy_unit_ids = frozenset(u.id for u in all_u[mid:])   # esquina superior
```

Las unidades con menor `x+y` (esquina superior-izquierda en StarCraft) pertenecen al equipo del agente (jugador 0). El set de IDs se fija en el primer frame del episodio y se mantiene durante todo el episodio: cuando una unidad muere simplemente desaparece de `state.units` y su ID deja de aparecer en `_classify()`.

### Formato de comando TorchCraft vs BWAPI

Un error crítico de desarrollo fue confundir los códigos de nivel TorchCraft con los tipos de comando BWAPI:

| Capa | Campo | Valor | Significado |
|---|---|---|---|
| **TorchCraft** (game-level) | `Command.code` | 0 | noop — no ejecutar nada |
| **TorchCraft** (game-level) | `Command.code` | 1 | **QUIT** — termina la partida |
| **TorchCraft** (game-level) | `Command.code` | 21 | `COMMAND_UNIT_PROTECTED` — ejecutar comando BWAPI en unidad propia |
| **BWAPI** (dentro de args) | `args[1]` | 1 | `UnitCommandType::Attack_Move` |
| **BWAPI** (dentro de args) | `args[1]` | 15 | `UnitCommandType::Gather` |

El formato correcto de un comando ATTACK_MOVE es `[21, uid, 1, -1, x, y, 0]`, donde `21` es el código TorchCraft que indica "ejecutar comando BWAPI" y `1` es el UnitCommandType de BWAPI dentro de los argumentos.

---

## 5. Protocolo TorchCraft

El paquete Python `torchcraft` fue eliminado de PyPI y requiere compilación con MSVC en Windows — inviable sin un entorno de build específico. El proyecto incluye una **reimplementación completa en Python puro** en `sc1_rl/torchcraft/proto.py` usando `pyzmq` y `flatbuffers`.

### 5.1 Formato de mensaje FlatBuffers

Todos los mensajes del protocolo TorchCraft v30 siguen la estructura:

```
Message {
  msg:  Any   (union discriminant en VT4, tabla en VT6)
  uid:  string (VT8)
}
```

El orden de campos en la VTable es **crítico**: `msg` (union) ocupa los slots 0 y 1 (VT4=msg_type ubyte, VT6=offset a la tabla interna), y `uid` ocupa el slot 2 (VT8). Una inversión de este orden provoca que BWEnv.dll reciba un `msg_type` incorrecto y descarte silenciosamente el mensaje.

### 5.2 Tipos de mensaje (`Any` enum)

| Valor | Nombre | Dirección | Descripción |
|---|---|---|---|
| 1 | `HandshakeClient` | → servidor | Versión de protocolo (30), nombre de UID |
| 2 | `Commands` | → servidor | Lista de comandos BWAPI para el frame actual |
| 3 | `HandshakeServer` | ← servidor | Nombre del mapa, player_id, dimensiones |
| 4 | `StateUpdate` | ← servidor | Frame completo o FrameDiff con todas las unidades |
| 5 | `PlayerLeft` | ← servidor | Un jugador abandonó la partida |
| 6 | `EndGame` | ← servidor | La partida terminó |
| 7 | `Error` | ← servidor | Error en el servidor |

### 5.3 Formato de comandos BWAPI

Cada comando enviado en `Commands{[Command]}` tiene la estructura:

```
Command {
  code: int32   (BWAPI::UnitCommandType::Enum)
  args: [int32] (unit_id, target_id, x, y, extra)
  str:  string  (vacío para la mayoría)
}
```

Los valores de `BWAPI::UnitCommandType::Enum` (versión 4.x) relevantes:

| Comando | Valor | Uso |
|---|---|---|
| `Attack_Move` | 1 | Mover y atacar hacia una posición |
| `Build` | 2 | Construir edificio |
| `Train` | 4 | Entrenar unidad |
| `Move` | 10 | Mover sin atacar |
| `Gather` | 15 | Recolectar mineral/gas |
| `Right_Click_Position` | 30 | Click derecho en posición |

---

## 6. Espacio de Observación

El `StateEncoder` convierte cada `GameState` en un vector `float32` de **189 características**:

| Rango | Categoría | Nº features | Descripción |
|---|---|---|---|
| [0:4] | Recursos | 4 | minerals, gas, supply_used, supply_max (normalizados a [0,1]) |
| [4:44] | Workers propios | 8 × 5 = 40 | x, y, hp_norm, is_idle, is_gathering |
| [44:104] | Armada propia | 12 × 5 = 60 | x, y, hp_norm, type_norm, is_attacking |
| [104:144] | Edificios propios | 8 × 5 = 40 | x, y, hp_norm, type_norm, is_training |
| [144:184] | Enemigos visibles | 10 × 4 = 40 | x, y, hp_norm, type_norm |
| [184:189] | Resumen global | 5 | n_workers, n_army, n_buildings, n_enemies, frame_norm |

Las coordenadas se normalizan dividiendo entre el tamaño del mapa en píxeles (`map_w × 8`, `map_h × 8`). Para el mapa `m5v5_c_far.scm`, en la práctica solo se usan las secciones de armada propia y enemigos (sin workers ni edificios).

---

## 7. Espacio de Acciones

Espacio discreto con **65 acciones** (ajustado al escenario de micro-combate puro):

| Rango | Tipo | Cantidad | Descripción |
|---|---|---|---|
| 0 | NOOP | 1 | Sin acción — avanza un frame sin comandos |
| 1–64 | ATTACK_MOVE | 64 | Ataque en grid 8×8 sobre el mapa completo |

El grid 8×8 divide el mapa en 64 celdas. La celda (row, col) se convierte a coordenadas en píxeles como el centroide de la celda:

```python
x = int((col + 0.5) / 8 * map_w_px)
y = int((row + 0.5) / 8 * map_h_px)
```

Para ATTACK_MOVE, el `CommandExecutor` envía el comando únicamente a las **unidades del equipo propio** (identificadas por split posicional), generando un comando `[TC_CMD_UNIT_PROTECTED=21, uid, CMD_ATTACK_MOVE=1, -1, x, y, 0]` por unidad. Las acciones de economía (construir, entrenar, recolectar) fueron eliminadas al migrar al escenario de combate puro `dragoons_zealots.scm`, donde no hay recursos ni producción.

---

## 8. Algoritmo de Aprendizaje — PPO

### 8.1 Red Neuronal (ActorCriticMLP)

```
Input: vector float32 (189 features)
  │
  ├── Linear(189 → 256) + LayerNorm + ReLU
  ├── Linear(256 → 256) + LayerNorm + ReLU
  │
  ├── Actor:  Linear(256 → 196) → logits → Categorical → acción discreta
  └── Critic: Linear(256 → 1)  → valor del estado V(s)
```

Se usa `LayerNorm` en lugar de `BatchNorm` para estabilidad con batch pequeño durante la recolección de rollouts (batch_size=1 durante el rollout, 64 durante el update).

Inicialización ortogonal con ganancia `√2` para capas ocultas; `0.01` para el actor (entropía alta inicial = exploración uniforme); `1.0` para el crítico.

### 8.2 Bucle de Entrenamiento

```
while timesteps < total_timesteps:
    # Fase 1: Recolección
    for t in range(rollout_steps=2048):
        obs → policy → action, log_prob, value
        env.step(action) → next_obs, reward, done
        buffer.store(obs, action, reward, value, log_prob)
    
    # Calcular ventajas GAE
    last_value = critic(last_obs)
    advantages = GAE(rewards, values, last_value, γ=0.99, λ=0.95)
    returns = advantages + values
    
    # Fase 2: Optimización PPO
    for epoch in range(4):
        for mini_batch in shuffle(buffer, size=64):
            ratio = exp(new_log_prob - old_log_prob)
            L_clip = min(ratio × A, clip(ratio, 1-ε, 1+ε) × A)
            L_vf   = (V_pred - V_target)²
            H      = -∑ π log π   (entropía)
            loss   = -L_clip + 0.5 × L_vf - 0.01 × H
            optimizer.step()
```

### 8.3 Hiperparámetros

| Parámetro | Valor |
|---|---|
| Tasa de aprendizaje | 3 × 10⁻⁴ |
| γ (descuento) | 0.99 |
| λ GAE | 0.95 |
| ε clipping | 0.2 |
| Épocas PPO por rollout | 4 |
| Tamaño de mini-batch | 64 |
| Pasos por rollout | 2 048 |
| Coeficiente de valor | 0.5 |
| Coeficiente de entropía | 0.01 |
| Norma máxima de gradiente | 0.5 |
| Pasos totales | 1 000 000 |
| Pasos máximos por episodio | 50 000 |

---

## 9. Recompensa

`TCRewardCalculator` calcula la recompensa por frame a partir del `GameState` estructurado:

| Señal | Valor | Descripción |
|---|---|---|
| Supervivencia | +0.001 / frame | Incentivo por mantenerse vivo |
| Acción ATTACK_MOVE | +0.005 / frame | Incentivo por enviar comandos activos |
| Penalización NOOP | −0.003 / frame | Desincentiva la inacción |
| Daño infligido | +0.001 × HP_perdido | Daño total causado al enemigo en el frame |
| Baja enemiga | +0.50 por baja | Unidad enemiga eliminada |
| Baja propia | −0.50 por baja | Unidad propia eliminada |
| Acercamiento al enemigo | +0.0005 × Δdist | Mejora en distancia media al enemigo más cercano |
| Victoria | +10.0 | Todas las unidades enemigas eliminadas |
| Derrota | −5.0 | Todas las unidades propias eliminadas |

La identificación de equipos para el cálculo de recompensa usa el mismo split posicional que el `StateEncoder` y el `CommandExecutor`. Los tres componentes inicializan `_own_unit_ids` de forma independiente pero con la misma lógica en el primer frame de cada episodio, y se resetean al inicio de cada episodio mediante `reward.reset()` y `executor.reset()`.

En una política aleatoria (inicio del entrenamiento), la recompensa esperada por episodio de 2000 pasos es ≈12.0 = 2000 × (attack_reward + survival) = 2000 × 0.006, sin kills ni muertes porque las unidades se mueven aleatoriamente sin converger al enemigo. La función de recompensa empieza a diferenciarse cuando el agente aprende a dirigir las unidades hacia el enemigo.

---

## 10. Errores de Desarrollo y Soluciones

Esta sección documenta cronológicamente los problemas encontrados durante la implementación y puesta en marcha del sistema. Se incluye como referencia para trabajos futuros y como evidencia del proceso de depuración.

---

### Error 1: DirectDraw 256 colores no disponible en VirtualBox

**Síntoma:** StarCraft arranca pero la pantalla aparece en negro o con colores completamente distorsionados.

**Causa:** StarCraft BW usa el modo gráfico DirectDraw con paleta de 256 colores. VirtualBox VBoxVGA no emula este modo de forma nativa.

**Solución:** Instalar `cnc-ddraw`, un wrapper de DirectDraw que intercepta las llamadas de la aplicación y las traduce al subsistema gráfico moderno. Una vez instalado en la carpeta de StarCraft, el juego arranca con colores correctos.

---

### Error 2: bwheadless no compatible con modo SINGLE_PLAYER

**Síntoma:** `bwheadless.exe -e StarCraft.exe -l bwapi-data\BWAPI.dll` lanza el juego pero auto_menu no puede crear una partida Single Player. El juego llega al menú principal y se detiene.

**Causa:** `bwheadless.exe` está diseñado para modo LAN multijugador. Su flag `--lan` inicia StarCraft en modo LAN, que requiere una partida de red. El modo SINGLE_PLAYER de `auto_menu` no funciona con esta configuración porque la secuencia de navegación de menús es diferente.

**Solución:** Usar **ChaosLauncher** con el plugin `BWAPI Injector [RELEASE]` en lugar de bwheadless. ChaosLauncher inyecta BWAPI directamente en StarCraft sin cambiar el modo de red, permitiendo que `auto_menu = SINGLE_PLAYER` funcione correctamente.

---

### Error 3: `game_type = MELEE` bloquea la partida en carga

**Síntoma:** La partida comienza a cargar (aparece la pantalla de carga con el mapa) pero se queda congelada indefinidamente. BWEnv.dll nunca envía el HandshakeServer.

**Causa:** Los mapas de micro-management de TorchCraft (`m5v5_c_far.scm`, etc.) están diseñados para el modo `USE_MAP_SETTINGS`. En este modo, las unidades están pre-colocadas en el mapa por el editor. El modo `MELEE` espera encontrar *starting locations* donde colocar las bases iniciales de cada jugador; estos mapas no tienen starting locations, por lo que StarCraft no puede inicializar la partida y se congela.

**Solución:** Cambiar en `bwapi.ini`:
```ini
game_type = USE_MAP_SETTINGS
```

---

### Error 4: `assume_on = false` congela StarCraft al iniciar

**Síntoma:** StarCraft arranca con ChaosLauncher y BWAPI inyectado, pero la pantalla se congela en los primeros segundos sin llegar al menú principal.

**Causa:** La opción `assume_on = false` en `torchcraft.ini` indica a BWEnv.dll que debe lanzar StarCraft por sí mismo (usando el launcher configurado, en este caso `bwheadless`). Cuando StarCraft ya está corriendo (lanzado por ChaosLauncher), BWEnv intenta lanzar *otro* proceso de StarCraft desde dentro del proceso ya en ejecución, lo cual resulta en un deadlock.

**Solución:** Cambiar en `torchcraft.ini`:
```ini
assume_on = true
```
Esto indica a BWEnv que StarCraft ya está corriendo y que solo debe inicializar el servidor ZMQ.

---

### Error 5: `torchcraft.ini` en ubicación incorrecta

**Síntoma:** BWEnv.dll no respeta las configuraciones de `torchcraft.ini` (puerto, log_path, assume_on). El comportamiento es el del archivo de configuración por defecto aunque el archivo exista en `C:\tc_data\`.

**Causa:** BWEnv.dll busca `torchcraft.ini` en una ruta específica relativa al directorio de StarCraft. El código fuente de BWEnv confirma que busca en `bwapi-data/torchcraft.ini` (relativo al directorio de StarCraft) y en `C:/StarCraft/bwapi-data/torchcraft.ini` (ruta hardcodeada de la instalación estándar). Colocar el archivo en `C:\tc_data\` (donde se guardan los logs) no tiene efecto.

**Solución:** Copiar `torchcraft.ini` a `C:\Archivos de programa\Starcraft\bwapi-data\torchcraft.ini`.

---

### Error 6: Orden de campos FlatBuffers incorrecto — BWEnv nunca responde al HandshakeClient

**Síntoma:** El cliente Python envía el HandshakeClient (72 bytes, PROTOCOL_VERSION=30) y espera 15–120 segundos sin recibir respuesta. BWEnv.dll tiene el puerto 11111 en estado LISTENING, pero nunca genera una respuesta.

**Causa:** Esta fue la causa raíz más difícil de diagnosticar. El esquema FlatBuffers de TorchCraft para `Message` es:

```
table Message {
  msg:  Any;    // union — genera VT4 (msg_type ubyte) y VT6 (msg offset)
  uid:  string; // VT8
}
```

La implementación inicial de `_message()` en `proto.py` tenía los campos en orden incorrecto: `uid` en slot 0 (VT4) y `msg` en slots 1 y 2 (VT6, VT8). Esto provocaba que BWEnv.dll leyera `msg_type` desde VT4 y encontrase el primer byte del string `uid` en lugar del discriminante del union. El servidor recibía un `msg_type` completamente inválido y descartaba silenciosamente cada mensaje.

**Diagnóstico:** Se confirmó el orden correcto de campos leyendo el archivo de cabecera `BWEnv/include/zmq_server.h` del código fuente de TorchCraft v1.4.0, que confirma `PROTOCOL_VERSION = 30` y el esquema del Message.

**Solución:** Corregir `_message()` en `proto.py`:
```python
def _message(b, uid_off, msg_type, inner_off):
    b.StartObject(3)
    b.PrependUint8Slot(0, msg_type, 0)              # slot 0 → VT4: msg_type
    b.PrependUOffsetTRelativeSlot(1, inner_off, 0)  # slot 1 → VT6: msg
    b.PrependUOffsetTRelativeSlot(2, uid_off, 0)    # slot 2 → VT8: uid
    return b.EndObject()
```

---

### Error 7: Firewall de Windows bloqueando el puerto 11111 en la VM

**Síntoma:** Después de corregir el protocolo FlatBuffers, `zmq_test.py` sigue sin recibir respuesta. `netstat -ano` en la VM muestra el puerto 11111 en estado LISTENING pero nunca aparece ninguna conexión ESTABLISHED, ni siquiera al hacer `sock.connect()` desde el host.

**Causa:** El Firewall de Windows de la VM bloqueaba las conexiones TCP entrantes en el puerto 11111. La regla se añadió con:
```cmd
netsh advfirewall firewall add rule name="TorchCraft ZMQ" dir=in action=allow protocol=TCP localport=11111
```
El comando devolvió `Ok`, pero las conexiones seguían sin establecerse.

**Causa raíz adicional:** La regla de firewall se aplicó correctamente, pero el port-forwarding NAT de VirtualBox no estaba funcionando correctamente para conexiones desde el propio host (`127.0.0.1`). VirtualBox NAT está diseñado principalmente para que la VM acceda al exterior, no para que el host acceda a la VM a través de NAT.

**Solución:** Conectar directamente a la IP del adaptador bridge de la VM (`192.168.0.23:11111`) en lugar de usar el port-forwarding NAT (`127.0.0.1:11111`). El adaptador bridge aparece en la misma red local que el host, sin capas de NAT intermedias. Con esta configuración, la conexión se establece en milisegundos.

Cambio en `config.yaml`:
```yaml
torchcraft:
  host: "192.168.0.23"  # Bridge adapter IP — bypasses VirtualBox NAT
  port: 11111
```

---

### Error 8: Checkpoint de arquitectura antigua incompatible

**Síntoma:** Al iniciar `python main.py`, el modelo encuentra un checkpoint de un entrenamiento anterior con arquitectura CNN y lanza `RuntimeError: Unexpected key(s) in state_dict: "features.conv.0.weight", ...`.

**Causa:** El arquitectura fue cambiada de CNN (con capas convolucionales `features.conv.*`) a MLP. El método `load()` en `agent.py` solo filtraba capas con *tamaño diferente*, pero no capas con *nombres que no existen* en la arquitectura actual.

**Solución:** Actualizar `agent.load()` para filtrar también claves desconocidas:
```python
unknown    = [k for k in saved if k not in current]
mismatched = [k for k in saved if k in current and saved[k].shape != current[k].shape]
skipped    = set(unknown) | set(mismatched)
compatible = {k: v for k, v in saved.items() if k not in skipped}
current.update(compatible)
self.network.load_state_dict(current)
```

---

### Error 9: Comandos BWAPI con valores enteros incorrectos — unidades estáticas

**Síntoma:** El handshake y el bucle de juego funcionan correctamente. El agente envía acciones ATTACK_MOVE pero las unidades propias permanecen completamente inmóviles. Solo se mueven las unidades enemigas (controladas por la IA de StarCraft).

**Causa:** El paquete `torchcraft` no está instalado (no está en `requirements.txt`). El módulo `constants.py` usa valores enteros de fallback hardcodeados para los tipos de comando BWAPI. Estos valores eran **incorrectos**:

```python
# INCORRECTO (fallback original)
CMD_MOVE               = 6   # → BWAPI::Research (!)
CMD_ATTACK_MOVE        = 13  # → BWAPI::Stop    (!)
CMD_GATHER             = 7   # → BWAPI::Upgrade (!)
CMD_BUILD              = 5   # → BWAPI::Morph   (!)
```

Cuando el agente enviaba `CMD_ATTACK_MOVE = 13`, BWEnv.dll lo interpretaba como `UnitCommandType::Stop`, deteniendo a todas las unidades. Esto explica perfectamente el síntoma: las unidades reciben un comando Stop continuo y nunca se mueven.

Los valores correctos de `BWAPI::UnitCommandType::Enum` (versión 4.x):

| Constante | Valor incorrecto | Valor correcto | Comando BWAPI |
|---|---|---|---|
| `CMD_ATTACK_MOVE` | 13 (Stop) | **1** | Attack_Move |
| `CMD_BUILD` | 5 (Morph) | **2** | Build |
| `CMD_MOVE` | 6 (Research) | **10** | Move |
| `CMD_GATHER` | 7 (Upgrade) | **15** | Gather |
| `CMD_RIGHT_CLICK_POS` | 14 (Follow) | **30** | Right_Click_Position |
| `CMD_RIGHT_CLICK_UNIT` | 15 (Gather) | **31** | Right_Click_Unit |
| `CMD_TRAIN` | 4 | **4** | Train ✓ (único correcto) |

**Solución:** Corregir los valores enteros en `constants.py`.

---

### Error 10: Doble ciclo send/recv por paso del entorno

**Síntoma:** El bucle de entrenamiento funciona pero consume el doble de frames de juego por cada paso del agente.

**Causa:** `TorchCraftClient.send(commands)` realiza el ciclo completo REQ→REP (envía comandos y recibe el siguiente estado). `recv()` es un alias de `send([])`, es decir, envía comandos vacíos (NOOP) y recibe otro estado. El entorno llamaba a ambos en secuencia:

```python
self.tc.send(commands)  # ciclo 1: envía acción, recibe Frame N+1
ok = self.tc.recv()     # ciclo 2: envía NOOP,   recibe Frame N+2  ← incorrecto
```

Resultado: se consumían 2 frames de juego por cada paso de entrenamiento, y el estado que se usaba para calcular la recompensa correspondía al frame **después** del NOOP, no al frame que resultó de la acción.

**Solución:** Eliminar la llamada redundante a `recv()`:
```python
ok = self.tc.send(commands)  # un único ciclo: envía y recibe
state = self.tc.state
```

---

### Error 11: Código de comando TorchCraft `code=1` = QUIT, no Attack_Move

**Síntoma:** El handshake y el bucle de juego funcionan correctamente. Las unidades reciben comandos ATTACK_MOVE pero el juego termina exactamente después de 3 pasos, siempre, con el mensaje `TorchCraft: game ended (msg_type=6)` un milisegundo después del tercer envío de comandos.

**Causa:** Confusión entre dos capas de codificación distintas. El formato de un comando TorchCraft es `[code, arg1, arg2, ...]` donde `code` es un enum de nivel **TorchCraft** (no BWAPI):

| `code` TorchCraft | Significado |
|---|---|
| 0 | noop |
| **1** | **QUIT — termina la partida inmediatamente** |
| 21 | `COMMAND_UNIT_PROTECTED` — ejecutar `args[1]` como UnitCommandType de BWAPI |

La implementación usaba `CMD_ATTACK_MOVE = 1` (que es el valor correcto como `UnitCommandType` de BWAPI) directamente como `code` del comando TorchCraft. Al recibir `code=1`, BWEnv.dll interpretaba QUIT y cerraba la partida. Esto es consistente con el comportamiento observado: exactamente 3 pasos, partida terminada a los 3 ms.

**Diagnóstico:** El problema se dedujo por la consistencia de "exactamente 3 pasos" y el log `msg_type=6` (EndGame) recibido inmediatamente tras el tercer Commands. Se confirmó consultando el código fuente de `BWEnv.cpp` que lista los valores del enum de comandos TorchCraft.

**Solución:** Añadir la constante `TC_CMD_UNIT_PROTECTED = 21` y cambiar el formato:

```python
# ANTES (incorrecto — envía QUIT):
commands.append([CMD_ATTACK_MOVE, uid, -1, x, y, 0])
#                      ↑ = 1 = QUIT en nivel TorchCraft

# DESPUÉS (correcto):
TC_CMD_UNIT_PROTECTED = 21  # TorchCraft game-level command
commands.append([TC_CMD_UNIT_PROTECTED, uid, CMD_ATTACK_MOVE, -1, x, y, 0])
#                       ↑ = 21 = ejecutar BWAPI cmd       ↑ = 1 = Attack_Move
```

---

### Error 12: BWEnv reporta `player_id=0` para todas las unidades — split posicional

**Síntoma:** El log de debug muestra `units_by_player_id={0: 16}` — todas las unidades de ambos equipos aparecen bajo el mismo `player_id=0`. Es imposible distinguir equipos por identificador de jugador.

**Causa:** BWEnv.dll en su configuración de micro-scenarios reporta el campo `playerId` (VT66 de la VTable de `Unit` en el FlatBuffer) como 0 para todas las unidades, independientemente del jugador BWAPI real. Esto ocurre porque BWEnv en modo micro-scenario consolida el estado de ambos jugadores bajo el ID del cliente conectado (jugador 0).

**Diagnóstico:** Se confirmó leyendo el campo VT66 directamente en `proto.py` y observando que siempre devuelve 0 para las 16 unidades del mapa `dragoons_zealots.scm`.

**Solución:** Identificar equipos por **posición inicial en la diagonal** `x+y`. Al inicio de cada episodio, se ordenan todas las unidades por `x+y` y se divide por la mediana: la mitad inferior pertenece al equipo propio (esquina superior-izquierda del mapa), la mitad superior al equipo enemigo:

```python
all_u.sort(key=lambda u: u.x + u.y)
mid = len(all_u) // 2
_own_unit_ids   = frozenset(u.id for u in all_u[:mid])
_enemy_unit_ids = frozenset(u.id for u in all_u[mid:])
```

Esta lógica se implementa de forma idéntica e independiente en `TCRewardCalculator`, `StateEncoder` y `CommandExecutor`, reseteándose al inicio de cada episodio.

---

### Error 13: Tipo de unidad BWAPI = 101, no 65 (Zealot) ni 66 (Dragoon)

**Síntoma:** `ARMY_TYPES` incluye `UTYPE_DRAGOON=66` y `UTYPE_ZEALOT=65`, pero el log de debug muestra `type_counts={101: 16}` — todas las unidades tienen tipo 101. Ningún comando llegaba a ejecutarse porque ninguna unidad superaba el filtro por tipo.

**Causa:** El tipo BWAPI reportado por TorchCraft para las unidades del mapa `dragoons_zealots.scm` no coincide con los valores de la enumeración `UnitType::Enum` de la documentación estándar de BWAPI 4.x. El tipo 101 no aparece documentado como un tipo estándar de Protoss. La causa exacta es desconocida (posible discrepancia entre versiones de TorchCraft/BWAPI).

**Solución:** Eliminar el filtro por tipo en `CommandExecutor._attack_move()` y usar exclusivamente el split posicional por UID para determinar qué unidades son propias. Los filtros por tipo se mantienen únicamente para excluir edificios, workers y recursos (que tienen tipos conocidos y estables):

```python
# Solo excluir lo que definitivamente no es combate
if unit.type in BUILDING_TYPES or unit.type in WORKER_TYPES or unit.type in RESOURCE_TYPES:
    continue
# Identificación de equipo: por UID, no por tipo
if self._own_unit_ids and unit.id not in self._own_unit_ids:
    continue
```

---

### Error 14: Socket ZMQ REQ atascado en estado EFSM tras error de recv

**Síntoma:** Después de que se produce un error en la comunicación ZMQ, todos los pasos siguientes fallan instantáneamente con `TorchCraft send/recv error: Operation cannot be accomplished in current state`. Los episodios duran exactamente 1 paso, con `reward=0` y el bucle de entrenamiento entra en un estado de fallo permanente a miles de episodios por segundo.

**Causa:** El socket ZMQ REQ implementa una máquina de estados estricta: send → recv → send → recv. Si `recv()` falla (timeout o error de red), el socket queda en estado "awaiting reply" (EFSM = Error de Finite State Machine). En este estado, cualquier intento de `send()` también falla con EFSM. El método `reconnect()` original reutilizaba el mismo socket roto, por lo que tampoco podía enviar el HandshakeClient.

```
Socket REQ atascado:
  send() → EFSM error
  recv() → EFSM error
  send() → EFSM error  (loop infinito)
```

Adicionalmente, como el socket nunca completaba el ciclo, `game_ended` nunca se ponía a `True`, por lo que `reset()` no llamaba a `reconnect()` y el loop continuaba indefinidamente.

**Solución:** Modificar `reconnect()` para cerrar el socket roto y crear uno nuevo antes de intentar el handshake. Añadir detección de EFSM en `reset()` para forzar reconexión si `recv()` falla sin `game_ended`:

```python
def reconnect(self) -> bool:
    self._close_socket()          # cierra socket roto (linger=0)
    self._sock = self._ctx.socket(zmq.REQ)  # socket fresco
    self._sock.connect(f"tcp://{self.host}:{self.port}")
    # ... handshake normal ...

# En reset():
ok = self.tc.recv()
if not ok and not needs_reconnect:
    if self.tc.reconnect():       # fallback: reconectar si EFSM
        ok = self.tc.recv()
```

---

### Error 15: Episodios terminan antes de `game_ended` — estado no se limpia entre partidas

**Síntoma:** A partir del segundo episodio, el split posicional devuelve 13v13 en lugar de 8v8. El log muestra `TEAMS own_ids=[0, 1, 2, 12, 14, 15, 26, 27, ...]` — unidades de la partida anterior mezcladas con las de la nueva.

**Causa:** El entorno tiene dos condiciones de terminación:
1. `game_ended = True` (BWEnv envió `EndGame`)
2. `combat_over = True` (conteo de unidades llegó a 0)

Cuando `combat_over` se dispara antes de que BWEnv envíe `EndGame`, el episodio termina con `game_ended=False`. Al llamar a `reset()`, la condición `if state.game_ended` no se cumple, por lo que **`reconnect()` no se llama** y el estado del cliente conserva las unidades de la partida anterior. BWEnv con `auto_restart` añade entonces las nuevas unidades encima de las antiguas en la siguiente respuesta, resultando en un estado contaminado.

**Solución en dos partes:**

1. **Dreno de frames** en `step()`: cuando `combat_over=True` pero `game_ended=False`, enviar hasta 60 NOOPs adicionales hasta recibir el `EndGame` oficial:
```python
if combat_over and not state.game_ended:
    for _ in range(60):
        if not self.tc.send([]) or self.tc.state.game_ended:
            break
    if not self.tc.state.game_ended:
        self.tc.state.game_ended = True  # forzar si BWEnv no responde
```

2. **Limpieza de stale units** en `reset()`: guardar los IDs de las unidades actuales antes de reconectar, y eliminarlos del primer frame del nuevo juego. Fallback adicional: si el número de unidades supera el esperado (16 para 8v8), conservar solo las 16 con IDs más altos (las más recientes):
```python
stale_ids = {u.id for pid_u in state.units.values() for u in pid_u.values()}
# ... reconectar ...
# eliminar stale_ids del nuevo estado
# fallback: conservar solo los expected_unit_count más recientes
```

---

### Error 16: Executor comandaba unidades de ambos equipos

**Síntoma:** El log `ATTACK_MOVE → 15 units → uids=[1,2,3,...,15]` muestra que el agente enviaba comandos a 15 de las 16 unidades — ambos equipos. Visualmente en el juego, unidades de ambos colores recibían órdenes.

**Causa:** El `CommandExecutor` filtraba unidades por `player_id == own_pid`. Como todos los `player_id` son 0 (ver Error 12), `is_own=True` para todas las unidades, y todas recibían comandos. Solo se excluía `uid=0` por el filtro `if uid <= 0: continue`. Los comandos `TC_CMD_UNIT_PROTECTED=21` son ignorados por BWAPI para las unidades que no pertenecen al jugador 0 (comandar unidades enemigas), pero el *agente creía estar comandando a su propio equipo* y el split posicional de la recompensa clasificaba incorrectamente los equipos.

**Solución:** Añadir la misma lógica de split posicional en `CommandExecutor`, con su propio `_own_unit_ids` reseteado al inicio de cada episodio mediante `executor.reset()`:

```python
class CommandExecutor:
    def __init__(self):
        self._own_unit_ids = None

    def reset(self):
        self._own_unit_ids = None

    def _init_teams(self, state):
        all_u = [u for units in state.units.values() for u in units.values()
                 if u.type not in RESOURCE_TYPES and u.type not in BUILDING_TYPES]
        all_u.sort(key=lambda u: u.x + u.y)
        mid = len(all_u) // 2
        self._own_unit_ids = frozenset(u.id for u in all_u[:mid])
```

---

## 11. Uso Rápido

### Requisitos previos

```powershell
# Instalar dependencias Python
.\proyecto-env\Scripts\Activate.ps1
pip install -r requirements.txt
pip install pyzmq flatbuffers
```

### Arranque de la VM y StarCraft

```powershell
# 1. Arrancar la VM
vagrant up

# 2. Dentro de la VM (acceder por RDP o consola VirtualBox):
#    a) Ejecutar ChaosLauncher.exe como Administrador
#    b) Asegurarse de que el plugin "BWAPI Injector 4.4.0 [RELEASE]" está activo
#    c) Pulsar "Start" en ChaosLauncher → StarCraft arranca y navega hasta la partida
#    d) Esperar a que la partida cargue (aparecen los Marines en el mapa)
```

### Verificar conectividad

```powershell
# Desde el host, verificar que BWEnv responde
python zmq_test.py
# Salida esperada:
# Protocol version: 30
# RESPUESTA recibida: XXXX bytes
# msg_type=3  (esperado 3=HandshakeServer)
```

### Entrenar

```powershell
# Entrenamiento nuevo desde cero
python main.py --fresh

# Continuar desde el último checkpoint
python main.py

# Checkpoint específico
python main.py --resume checkpoints\step_00050000.pt

# Configuración alternativa
python main.py --config config_alt.yaml
```

### Estructura de archivos

```
Proyecto_Final/
├── main.py                        # Punto de entrada
├── config.yaml                    # Hiperparámetros y configuración
├── zmq_test.py                    # Diagnóstico de conexión TorchCraft
├── requirements.txt               # Dependencias Python
├── Vagrantfile                    # Configuración de la VM
│
├── sc1_rl/
│   ├── torchcraft/
│   │   ├── proto.py               # FlatBuffers TorchCraft puro Python
│   │   ├── client.py              # TorchCraftClient (ZMQ REQ)
│   │   ├── constants.py           # Constantes BWAPI (tipos de unidad y comando)
│   │   ├── action_space.py        # Definición de 196 macro-acciones
│   │   ├── command_executor.py    # Traductor TCAction → comandos BWAPI
│   │   ├── state_encoder.py       # GameState → vector float32 (189 features)
│   │   └── reward.py              # TCRewardCalculator
│   ├── environment/
│   │   └── sc1_env_tc.py          # gymnasium.Env sobre TorchCraftClient
│   ├── model/
│   │   ├── agent.py               # PPOAgent (policy + optimizer + load/save)
│   │   ├── network_mlp.py         # ActorCriticMLP (189 → 256 → 256 → actor/critic)
│   │   ├── trainer.py             # Bucle rollout → GAE → PPO update
│   │   └── memory.py              # RolloutBuffer con GAE
│   └── logger/
│       └── action_logger.py       # Logger de acciones y episodios
│
├── VBox/Local Disk/               # Snapshot del filesystem de la VM
│   └── Archivos de programa/Starcraft/
│       └── bwapi-data/
│           ├── bwapi.ini          # Configuración BWAPI
│           └── torchcraft.ini     # Configuración BWEnv.dll
│
├── checkpoints/                   # Checkpoints PPO guardados
├── logs/                          # Logs de entrenamiento
└── tc_maps/                       # Mapas de micro-management TorchCraft
```

---

## 12. Recursos Utilizados

### Software

| Recurso | Versión | Rol |
|---|---|---|
| Python | 3.13 | Runtime principal del agente |
| PyTorch | ≥ 2.1 | Red neuronal y backpropagation |
| Gymnasium | ≥ 0.29 | Interfaz estándar de entorno RL |
| NumPy | ≥ 1.26 | Operaciones matriciales en el buffer de rollout |
| pyzmq | ≥ 26 | Socket ZMQ para comunicación con BWEnv |
| flatbuffers | ≥ 24 | Serialización manual del protocolo TorchCraft |
| PyYAML | ≥ 6.0 | Carga de configuración |
| VirtualBox | 7.x | Hypervisor para la VM de juego |
| Vagrant | 2.x | Automatización del ciclo de vida de la VM |
| BWAPI | 4.4.0 | Hook en StarCraft BW — expone API del juego |
| TorchCraft | v1.4.0 | BWEnv.dll — servidor ZMQ de estado del juego |
| ChaosLauncher | — | Inyector de BWAPI en StarCraft |
| cnc-ddraw | — | Wrapper DirectDraw para compatibilidad de color |
| StarCraft BW | 1.16.1 | Entorno de juego |

### Hardware (host)

| Recurso | Uso |
|---|---|
| CPU | Bucle Python, encoding, rollout collection |
| GPU (CUDA) | Inferencia y entrenamiento PPO |
| RAM | ≥ 8 GB reservados para la VM + modelo |
| Red local | Comunicación ZMQ host↔VM por bridge (< 1 ms latencia) |

---

## 13. Trabajo Futuro

- **Convergencia de combate:** la señal de recompensa por distancia al enemigo (`+0.0005 × Δdist`) necesita refinamiento; el agente actual obtiene recompensa ~12 por episodio (solo survival + attack) sin llegar a matar unidades enemigas con política aleatoria. Explorar reward shaping más agresivo (*focus fire*, kiting explícito).
- **Sincronización de IDs de equipo:** los tres componentes (`StateEncoder`, `TCRewardCalculator`, `CommandExecutor`) mantienen su propio `_own_unit_ids` independiente. En el futuro convendría centralizar esta lógica en un objeto `TeamTracker` compartido para garantizar consistencia estricta.
- **Identificación de equipo por comportamiento:** en lugar del split posicional (que puede fallar si ambos equipos empiezan en posiciones similares), detectar el equipo propio observando qué unidades responden a los comandos `TC_CMD_UNIT_PROTECTED` en los primeros frames.
- **Self-play:** entrenar el agente contra versiones anteriores de sí mismo en lugar de contra la IA de StarCraft.
- **Escalado de escenario:** aumentar el tamaño del combate (10v10, 20v20) y diversificar las razas.
- **Arquitecturas recurrentes:** añadir LSTM para información parcialmente observable.
- **Estabilidad de reconexión:** el protocolo de auto_restart de BWEnv no limpia correctamente las unidades entre partidas (Error 15 y 16); una solución más robusta sería reiniciar BWEnv completamente entre episodios en lugar de depender de `auto_restart=ON`.

---

## 14. Referencias

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
- Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518(7540), 529–533.
- Synnaeve, G., et al. (2016). *TorchCraft: a Library for Machine Learning Research on Real-Time Strategy Games*. arXiv:1611.00625.
- Vinyals, O., et al. (2019). *Grandmaster level in StarCraft II using multi-agent reinforcement learning*. Nature, 575(7782), 350–354.
- BWAPI Development Team. *BWAPI 4.4.0 Documentation*. https://bwapi.github.io
- Flatbuffers Documentation. *Writing a Schema*. https://flatbuffers.dev/flatbuffers_guide_writing_schema.html
