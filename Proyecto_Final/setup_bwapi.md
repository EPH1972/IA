# Instalación de BWAPI + TorchCraft en la VM

Pasos para activar `python main.py` con el servidor TorchCraft.  
Todos los archivos necesarios están en `S:\torchcraft-v1.4.0\` (unidad compartida de la VM).

---

## 1. Requisitos previos (en el host)

```powershell
pip install torchcraft
```

---

## 2. Dentro de la VM — instalar BWAPI

Los instaladores están en la unidad compartida `S:\` (carpeta raíz del proyecto host).

> Si `S:` no aparece, abre el Explorador → `\\vboxsvr\sc1_files`.  
> Requiere VirtualBox Guest Additions instaladas.

1. Ejecuta `S:\BWAPI_440_Setup.exe` **como Administrador**.  
   El instalador detecta StarCraft automáticamente y crea la estructura:
   ```
   C:\Archivos de Programa\Starcraft\dbghelp.dll
   C:\Archivos de Programa\Starcraft\bwapi-data\
   C:\Archivos de Programa\Starcraft\bwapi-data\AI\
   C:\Archivos de Programa\Starcraft\bwapi-data\BWAPI.ini
   ```

---

## 3. Dentro de la VM — copiar archivos de TorchCraft

Extrae `S:\torchcraft-v1.4.0.zip` (o copia directamente desde `S:\torchcraft-v1.4.0\`):

| Origen en el zip | Destino en la VM |
|---|---|
| `torchcraft-v1.4.0/BWEnv.dll` | `C:\Archivos de Programa\Starcraft\bwapi-data\AI\BWEnv.dll` |
| `torchcraft-v1.4.0/bin/libzmq.dll` | `C:\Archivos de Programa\Starcraft\libzmq.dll` |
| `torchcraft-v1.4.0/bin/bwheadless.exe` | `C:\Archivos de Programa\Starcraft\bwheadless.exe` |
| `torchcraft-v1.4.0/config/torchcraft.ini` | `C:\Archivos de Programa\Starcraft\bwapi-data\torchcraft.ini` |

---

## 4. Configurar BWAPI.ini

Reemplaza `C:\Archivos de Programa\Starcraft\bwapi-data\BWAPI.ini` con el del zip
(`torchcraft-v1.4.0/config/bwapi.ini`) y ajusta solo estas líneas:

```ini
[ai]
ai     = bwapi-data\AI\BWEnv.dll
ai_dbg = bwapi-data\AI\BWEnv.dll

[auto_menu]
auto_menu  = SINGLE_PLAYER
race       = Terran
enemy_race = Random
enemy_count = 1
map        = maps\(2)Heartbreak Ridge.scm   ; o cualquier mapa 1v1 disponible

[starcraft]
; Slowest speed (0 ms/frame = máxima velocidad, ajustar según preferencia)
;speed_override = -1
```

> El resto del `.ini` del zip ya viene correctamente configurado para TorchCraft.

---

## 5. Iniciar StarCraft headless con bwheadless

`bwheadless.exe` arranca StarCraft sin ventana, con BWAPI inyectado.
Abre un **símbolo del sistema como Administrador** y ejecuta:

```cmd
cd "C:\Archivos de Programa\Starcraft"
bwheadless.exe -e StarCraft.exe -l "bwapi-data\BWAPI.dll"
```

StarCraft arranca en segundo plano, BWAPI carga `BWEnv.dll` y el servidor ZMQ
queda escuchando en el puerto **11111**.

> Si bwheadless no encuentra `BWAPI.dll`, usa la ruta completa:
> ```cmd
> bwheadless.exe -e "C:\Archivos de Programa\Starcraft\StarCraft.exe" -l "C:\Archivos de Programa\Starcraft\bwapi-data\BWAPI.dll"
> ```

---

## 6. Lanzar el entrenamiento (en el host)

```powershell
# Activa el entorno Python
.\proyecto-env\Scripts\Activate.ps1

# Comprueba que el servidor TorchCraft responde
Test-NetConnection -ComputerName 127.0.0.1 -Port 11111
# TcpTestSucceeded : True  →  listo

# Entrena
python main.py --fresh
```

---

## Arquitectura de arranque

```
Host Windows 10
│
├── python main.py
│     └── TorchCraftClient  ──ZMQ──►  127.0.0.1:11111
│                                         (port-forward Vagrantfile)
└── VirtualBox VM (Win7)
      └── bwheadless.exe
            └── StarCraft.exe (headless)
                  └── BWAPI 4.4  →  BWEnv.dll  →  ZMQ server :11111
```

---

## Contenido del zip (referencia)

```
torchcraft-v1.4.0/
  BWEnv.dll                ← DLL principal (va en bwapi-data/AI/)
  bin/
    bwheadless.exe          ← lanzador headless de StarCraft
    libzmq.dll              ← biblioteca ZMQ (va en C:\Archivos de Programa\Starcraft\)
  config/
    bwapi.ini               ← configuración BWAPI para TorchCraft
    torchcraft.ini          ← puerto, modo imagen, launcher
  maps/                     ← mapas de micro opcionales
```
