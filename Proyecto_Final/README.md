# Aprendizaje por Refuerzo en StarCraft: Brood War mediante TorchCraft y PPO

**Autor:** Eduard Pérez  
**Fecha:** Mayo 2026  
**Estado:** Entrenamiento activo

---

## Resumen

Este trabajo presenta el diseño, implementación y puesta en marcha de un agente de aprendizaje por refuerzo (RL) capaz de jugar a *StarCraft: Brood War 1.16.1* en un escenario de micro-management de combate. El agente se ejecuta en un host Windows 10 y se comunica con el juego a través del protocolo TorchCraft ZMQ, que expone el estado estructurado del juego (posiciones, HP, tipos de unidad) mediante BWAPI 4.4.0 inyectado en el proceso de StarCraft dentro de una máquina virtual Windows 7. El algoritmo de aprendizaje es **Proximal Policy Optimization (PPO)** con una red MLP que consume un vector de observación de 189 características. El cliente TorchCraft es una reimplementación completa en Python puro con FlatBuffers manual, prescindiendo de la extensión C que ya no está disponible en PyPI.

---

## Índice

1. [Introducción](#1-introducción)
2. [Entorno de Ejecución](#2-entorno-de-ejecución)
3. [Pila de Software dentro de la VM](#3-pila-de-software-dentro-de-la-vm)
4. [Arquitectura del Sistema](#4-arquitectura-del-sistema)
5. [Protocolo TorchCraft](#5-protocolo-torchcraft)
6. [Espacio de Observación](#6-espacio-de-observación)
7. [Espacio de Acciones](#7-espacio-de-acciones)
8. [Algoritmo de Aprendizaje — PPO](#8-algoritmo-de-aprendizaje--ppo)
9. [Recompensa](#9-recompensa)
10. [Errores de Desarrollo y Soluciones](#10-errores-de-desarrollo-y-soluciones)
11. [Uso Rápido](#11-uso-rápido)
12. [Recursos Utilizados](#12-recursos-utilizados)
13. [Trabajo Futuro](#13-trabajo-futuro)
14. [Referencias](#14-referencias)

---

## 1. Introducción

*StarCraft: Brood War* es uno de los entornos de referencia más complejos para agentes artificiales. Su espacio de acciones masivo, la información imperfecta y la necesidad de micro-gestión táctica en tiempo real lo convierten en un banco de pruebas ideal para algoritmos de RL modernos. Este proyecto se centra en el problema de **micro-management de combate**: controlar un grupo de Marines Terran para derrotar a un grupo enemigo equivalente en el menor número de frames posible.

El escenario elegido es el mapa `m5v5_c_far.scm` (5 Marines vs. 5 Zerglings, posiciones alejadas), un benchmark estándar de la comunidad TorchCraft. El problema de micro-management ya es suficientemente rico para estudiar comportamientos emergentes como *focus fire*, *kiting* y posicionamiento táctico, sin la complejidad adicional de economía y construcción que introduce el juego completo.

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
map          = Maps/BroodWar/micro/m5v5_c_far.scm
race         = Terran
enemy_count  = 1
enemy_race   = Zerg
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
│  Mapa: Maps/BroodWar/micro/m5v5_c_far.scm        │
│  5 Marines Terran  vs.  5 Zerglings Zerg         │
└──────────────────────────────────────────────────┘
```

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

Espacio discreto con **196 acciones**:

| Rango | Tipo | Cantidad | Descripción |
|---|---|---|---|
| 0 | NOOP | 1 | Sin acción — avanza un frame sin comandos |
| 1 | GATHER_IDLE_WORKERS | 1 | Asigna todos los SCVs ociosos al recurso más cercano |
| 2–65 | ATTACK_MOVE | 64 | Ataque en grid 8×8 sobre el mapa completo |
| 66–129 | BUILD_SUPPLY_DEPOT | 64 | Construir Supply Depot en grid 8×8 |
| 130–193 | BUILD_BARRACKS | 64 | Construir Barracks en grid 8×8 |
| 194 | TRAIN_SCV | 1 | Entrenar SCV desde el Command Center |
| 195 | TRAIN_MARINE | 1 | Entrenar Marine desde cualquier Barracks |

El grid 8×8 divide el mapa en 64 celdas. Para ATTACK_MOVE, el `CommandExecutor` envía a **todas** las unidades de combate hacia la celda seleccionada (posición en píxeles del centroide de la celda).

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
| HP infligido (× 0.01) | proporcional | Daño total causado al enemigo en el frame |
| HP recibido (× −0.005) | proporcional | Daño recibido por las propias unidades |
| Bajas enemigas (× 1.0) | +1.0 por baja | Unidad enemiga eliminada |
| Bajas propias (× −1.0) | −1.0 por baja | Unidad propia eliminada |
| Victoria | +10.0 | Todas las unidades enemigas eliminadas |
| Derrota | −10.0 | Todas las unidades propias eliminadas |
| Supervivencia | +0.001 / frame | Incentivo mínimo para no quedarse inmóvil |

La función de recompensa se orienta a combate puro: maximizar el daño infligido y las bajas enemigas mientras se minimizan las propias bajas. Esto es apropiado para el mapa `m5v5_c_far.scm` donde no hay economía ni construcción.

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

- **Reward shaping:** diseñar señales de recompensa intermedias basadas en distancia al enemigo, concentración de fuego (*focus fire*) y kiting para acelerar el aprendizaje en los primeros episodios.
- **Self-play:** entrenar el agente contra versiones anteriores de sí mismo en lugar de contra la IA de StarCraft, eliminando el sesgo hacia un único estilo de juego enemigo.
- **Escalado de escenario:** aumentar el tamaño del combate (10v10, 20v20) y diversificar las razas enemigas para generalización.
- **Arquitecturas recurrentes:** añadir una capa LSTM entre el encoder y la política para manejar información parcialmente observable (unidades fuera de rango visual).
- **Juego completo:** como extensión a largo plazo, ampliar el espacio de acciones y observación para cubrir economía, construcción y producción de unidades, acercándose al problema original de AlphaStar.

---

## 14. Referencias

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
- Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518(7540), 529–533.
- Synnaeve, G., et al. (2016). *TorchCraft: a Library for Machine Learning Research on Real-Time Strategy Games*. arXiv:1611.00625.
- Vinyals, O., et al. (2019). *Grandmaster level in StarCraft II using multi-agent reinforcement learning*. Nature, 575(7782), 350–354.
- BWAPI Development Team. *BWAPI 4.4.0 Documentation*. https://bwapi.github.io
- Flatbuffers Documentation. *Writing a Schema*. https://flatbuffers.dev/flatbuffers_guide_writing_schema.html
