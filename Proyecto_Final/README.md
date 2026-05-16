# Aprendizaje por Refuerzo Aplicado a StarCraft: Brood War mediante Visión por Computador y Control Remoto de Máquina Virtual

**Autor:** Eduard Pérez  
**Fecha de inicio:** Mayo 2026  
**Estado:** En desarrollo

---

## Resumen

Este trabajo presenta el diseño e implementación de un agente de aprendizaje por refuerzo (RL) capaz de jugar a *StarCraft: Brood War 1.16.1* sin ninguna modificación del ejecutable del juego ni acceso a su estado interno. El agente observa el juego exclusivamente a través de capturas de pantalla de una máquina virtual Windows 7 y actúa sobre ella enviando comandos de teclado y ratón mediante la interfaz de línea de comandos de VirtualBox (`VBoxManage`). El algoritmo de aprendizaje empleado es Proximal Policy Optimization (PPO) con una red neuronal convolucional como extractor de características.

---

## 1. Introducción

*StarCraft: Brood War* es considerado uno de los entornos de referencia más complejos para agentes artificiales, debido a su espacio de acciones masivo, su naturaleza de información imperfecta y la necesidad de planificación a largo plazo. Trabajos previos como TorchCraft (Synnaeve et al., 2016) o AlphaStar (Vinyals et al., 2019) han atacado este problema mediante APIs que exponen directamente el estado del juego. El presente proyecto adopta una perspectiva diferente: el agente aprende **exclusivamente desde píxeles**, operando sobre una máquina virtual de la misma forma que lo haría un jugador humano, sin hooks en el proceso del juego.

El principal desafío técnico es la capa de comunicación: el juego corre en una VM Windows 7 aislada (VirtualBox + Vagrant) y el agente reside en el host. No existe ningún canal de comunicación privilegiado; toda interacción se realiza a través de las herramientas de administración de VirtualBox.

---

## 2. Entorno de Ejecución

### 2.1 Máquina Virtual

| Parámetro | Valor |
|---|---|
| Hypervisor | Oracle VirtualBox (Vagrant) |
| Sistema operativo invitado | Windows 7 Lite (sin conexión) |
| RAM asignada | 4 GB |
| CPUs asignadas | 2 |
| Controlador gráfico | VBoxVGA (compatible con DirectDraw) |
| VRAM | 128 MB |
| Resolución de juego | 640 × 480 px |
| Versión de StarCraft | Brood War 1.16.1 |

La VM se gestiona con Vagrant (`vagrant up / halt / destroy`). No se habilita SSH ni WinRM; la comunicación es exclusivamente a través de `VBoxManage`.

### 2.2 Host

| Parámetro | Valor |
|---|---|
| Sistema operativo | Windows 10 Pro (10.0.19045) |
| Runtime Python | Python 3.13.3 |
| Entorno virtual | `proyecto-env/` (venv) |
| Framework RL | PyTorch 2.12 + Gymnasium 1.3 |

---

## 3. Arquitectura del Sistema

El sistema se divide en dos secciones con responsabilidades estrictamente separadas.

```
┌──────────────────────────────────────────────────────┐
│                     HOST (Python)                    │
│                                                      │
│  ┌──────────────── SEC 1: MODELO ─────────────────┐  │
│  │                                                │  │
│  │   SC1Env (Gymnasium)                           │  │
│  │     ├── StateProcessor  → frame stack [0,1]   │  │
│  │     ├── RewardCalculator → recompensa visual  │  │
│  │     └── ActionLogger    → log JSONL + texto   │  │
│  │                                                │  │
│  │   PPOAgent                                     │  │
│  │     ├── ActorCritic (CNN)                      │  │
│  │     └── RolloutBuffer (GAE)                    │  │
│  │                                                │  │
│  │   Trainer → bucle principal                    │  │
│  └────────────────────┬───────────────────────────┘  │
│                       │ Action / Screenshot           │
│  ┌──────────── SEC 2: COMUNICACIÓN VM ────────────┐  │
│  │                                                │  │
│  │   VMController                                 │  │
│  │     ├── VMConnector   → start / stop / estado  │  │
│  │     ├── ScreenCapture → screenshotpng → PIL    │  │
│  │     └── InputHandler  → scan codes + mouse abs │  │
│  └────────────────────┬───────────────────────────┘  │
│                       │ VBoxManage CLI               │
└───────────────────────┼──────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────┐
│              VM Windows 7 — StarCraft BW              │
└──────────────────────────────────────────────────────┘
```

### 3.1 SEC 1 — Modelo

**`SC1Env`** implementa la interfaz `gymnasium.Env`. En cada paso:
1. Registra la acción en el log antes de enviarla (trazabilidad completa).
2. Delega la ejecución al `VMController` (SEC 2).
3. Captura la nueva pantalla y la procesa con `StateProcessor`.
4. Calcula la recompensa con `RewardCalculator`.

**`StateProcessor`** convierte cada screenshot PIL en un array `float32` de forma `(4, 128, 128)`: escala de grises, redimensionado bilineal y apilado de los últimos 4 frames (frame stacking para percepción de movimiento).

**`RewardCalculator`** usa una señal de recompensa puramente visual:
- Bonus de supervivencia: `+0.001` por paso.
- Penalización de NOOP: `−0.002` por acción nula.
- Actividad de pantalla: `diff_frames × 0.1` (incentiva que el juego avance).

**`ActionLogger`** escribe en paralelo en dos canales:
- `logs/training.log` — texto legible con timestamps.
- `logs/action_log.jsonl` — una línea JSON por cada acción enviada a la VM, con episodio, paso, tipo de acción y coordenadas.

### 3.2 SEC 2 — Comunicación con la VM

Toda la comunicación con la VM se realiza exclusivamente mediante `VBoxManage`, sin instalar software adicional en el guest.

| Componente | Mecanismo VBoxManage | Descripción |
|---|---|---|
| `ScreenCapture` | `controlvm <vm> screenshotpng <ruta>` | Captura el framebuffer actual |
| `InputHandler` (teclado) | `controlvm <vm> keyboardputscancode <hex...>` | AT scan codes: press + release |
| `InputHandler` (ratón) | `controlvm <vm> mouse abs x y 0 0 buttons` | Coordenadas absolutas en píxeles |
| `VMConnector` | `startvm`, `showvminfo`, `controlvm poweroff` | Ciclo de vida de la VM |

---

## 4. Espacio de Acciones

El espacio de acciones es discreto con **548 acciones** totales, distribuidas en:

| Categoría | Cantidad | Descripción |
|---|---|---|
| NOOP | 1 | Sin acción |
| Cámara | 4 | Flechas arriba / abajo / izquierda / derecha |
| Left click | 256 | Grid 16×16 sobre la pantalla completa |
| Right click | 256 | Grid 16×16 sobre la pantalla completa |
| Hotkeys | 13 | A, B, S, H, P, U, R, T, M, G, Escape, Enter, F10 |
| Seleccionar grupo | 9 | Teclas 1–9 |
| Asignar grupo | 9 | Ctrl + 1–9 |

Los clics de pantalla se mapean a un grid 16×16, lo que da una resolución de 40×30 píxeles por celda sobre la resolución de juego de 640×480.

---

## 5. Algoritmo de Aprendizaje — PPO

Se emplea **Proximal Policy Optimization** (Schulman et al., 2017) por su estabilidad en entornos de alta varianza y su eficiencia con rollouts on-policy.

### 5.1 Red Neuronal (ActorCritic)

```
Input: (4, 128, 128)  — 4 frames en escala de grises
  │
  ├── Conv2d(4→32,  kernel=8, stride=4) + ReLU
  ├── Conv2d(32→64, kernel=4, stride=2) + ReLU
  ├── Conv2d(64→64, kernel=3, stride=1) + ReLU
  ├── Flatten
  └── Linear(→512) + ReLU
        │
        ├── Actor:  Linear(512→548)  → logits → Categorical → acción
        └── Critic: Linear(512→1)   → valor del estado
```

Inicialización ortogonal con ganancia `sqrt(2)` para capas convolucionales y lineales ocultas; ganancia `0.01` para el actor y `1.0` para el crítico.

### 5.2 Hiperparámetros

| Parámetro | Valor |
|---|---|
| Tasa de aprendizaje | 3 × 10⁻⁴ |
| Factor de descuento γ | 0.99 |
| λ GAE | 0.95 |
| Clipping ε | 0.2 |
| Épocas PPO por rollout | 4 |
| Tamaño de mini-batch | 64 |
| Pasos por rollout | 2 048 |
| Coeficiente de valor | 0.5 |
| Coeficiente de entropía | 0.01 |
| Norma máxima de gradiente | 0.5 |

### 5.3 Función de pérdida

$$\mathcal{L} = \mathcal{L}^{CLIP} + 0.5 \cdot \mathcal{L}^{VF} - 0.01 \cdot \mathcal{H}$$

Donde:
- $\mathcal{L}^{CLIP}$: pérdida de política con clipping PPO.
- $\mathcal{L}^{VF}$: MSE entre el valor estimado y los retornos calculados con GAE.
- $\mathcal{H}$: entropía de la distribución de política (fomenta exploración).

---

## 6. Recursos Utilizados

### Software

| Recurso | Versión | Uso |
|---|---|---|
| Python | 3.13.3 | Runtime principal |
| PyTorch | 2.12.0 | Red neuronal y backpropagation |
| Gymnasium | 1.3.0 | Interfaz estándar de entorno RL |
| NumPy | 2.4.5 | Operaciones matriciales en el buffer |
| Pillow | 12.2.0 | Procesamiento de capturas de pantalla |
| PyYAML | 6.0.3 | Carga de configuración |
| VirtualBox | — | Hypervisor para la VM de juego |
| Vagrant | — | Automatización del ciclo de vida de la VM |

### Hardware (host)

| Recurso | Descripción |
|---|---|
| CPU | Usado para inferencia y comunicación con VBoxManage |
| GPU | Aceleración CUDA para entrenamiento (si disponible) |
| RAM | 4 GB reservados para la VM + memoria del modelo en host |

---

## 7. Problemas Encontrados

*Esta sección se irá completando a medida que surjan durante el desarrollo.*

---

## 8. Trabajo Futuro

- Incorporar señales de recompensa más ricas (reconocimiento óptico de la puntuación en pantalla).
- Evaluar arquitecturas recurrentes (LSTM) para manejar información parcialmente observable.
- Explorar un espacio de acciones jerárquico (selección de unidad → tipo de orden → destino).
- Comparar PPO con otros algoritmos on-policy (A2C) y off-policy (SAC discreto).

---

## 9. Referencias

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
- Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518(7540), 529–533.
- Synnaeve, G., et al. (2016). *TorchCraft: a Library for Machine Learning Research on Real-Time Strategy Games*. arXiv:1611.00625.
- Vinyals, O., et al. (2019). *Grandmaster level in StarCraft II using multi-agent reinforcement learning*. Nature, 575(7782), 350–354.

---

## Uso Rápido

```powershell
# 1. Arrancar la VM
vagrant up

# 2. Activar entorno Python
.\proyecto-env\Scripts\Activate.ps1

# 3. Entrenar
python main.py

# 4. Reanudar desde checkpoint
python main.py --resume checkpoints\step_00010000.pt

# 5. Apagar VM al terminar
vagrant halt
```
