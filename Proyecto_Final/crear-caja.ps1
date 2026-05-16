#Requires -Version 5.1
# crear-caja.ps1
# Genera la caja Vagrant local "win7-lite" a partir de VBoxManage.
# Ejecutar UNA VEZ antes del primer "vagrant up".

param([switch]$Force)

$ErrorActionPreference = "Stop"

$BoxName  = "win7-lite"
$VagrantD = if ($env:VAGRANT_HOME) { $env:VAGRANT_HOME } else { "$env:USERPROFILE\.vagrant.d" }
$BoxDir   = Join-Path $VagrantD "boxes\$BoxName\0\virtualbox"

if (-not $Force -and (Test-Path "$BoxDir\box.ovf")) {
    Write-Host "La caja '$BoxName' ya existe. Usa -Force para recrearla."
    exit 0
}

# Localizar VBoxManage
$cmd = Get-Command VBoxManage -ErrorAction SilentlyContinue
$vbm = if ($cmd) { $cmd.Source } else { $null }
if (-not $vbm) {
    foreach ($candidate in @(
        "C:\Program Files\Oracle\VirtualBox\VBoxManage.exe",
        "C:\Program Files (x86)\Oracle\VirtualBox\VBoxManage.exe"
    )) {
        if (Test-Path $candidate) { $vbm = $candidate; break }
    }
}
if (-not $vbm) { throw "VBoxManage no encontrado. Instala VirtualBox primero." }

# Ejecuta VBoxManage silenciando la barra de progreso en stderr (PS 5.1 la trata como error)
function vbm {
    $out = & $vbm @args 2>&1
    if ($LASTEXITCODE -ne 0) { throw "VBoxManage falló ($LASTEXITCODE): $out" }
}

$TempVm  = "vagrant-dummy-$(Get-Random)"
$TempDir = Join-Path $env:TEMP "vagrantbox-$(Get-Random)"
New-Item -ItemType Directory -Path $TempDir | Out-Null

Write-Host "Creando caja Vagrant '$BoxName'..."
try {
    # 1. VM temporal Windows 7 64-bit
    Write-Host "  [1/5] Registrando VM temporal..."
    vbm createvm --name $TempVm --ostype Windows7_64 --register

    # 2. Disco mínimo 1 MB (solo para que el OVF tenga referencia de disco)
    Write-Host "  [2/5] Creando disco mínimo..."
    $TempDisk = Join-Path $TempDir "dummy.vmdk"
    vbm createmedium disk --filename $TempDisk --size 1 --format VMDK --variant Fixed

    # 3. Controlador IDE + disco + ajustes mínimos
    Write-Host "  [3/5] Configurando almacenamiento..."
    vbm storagectl $TempVm --name "IDE Controller" --add ide --controller PIIX4
    vbm storageattach $TempVm --storagectl "IDE Controller" `
        --port 0 --device 0 --type hdd --medium $TempDisk
    vbm modifyvm $TempVm --memory 512 --vram 16

    # 4. Exportar como OVF (Vagrant requiere que el OVF se llame box.ovf)
    Write-Host "  [4/5] Exportando OVF..."
    $OvfOut = Join-Path $TempDir "box.ovf"
    vbm export $TempVm --output $OvfOut

    # 5. Copiar al directorio de la caja Vagrant
    Write-Host "  [5/5] Instalando caja..."
    if ($Force -and (Test-Path $BoxDir)) { Remove-Item -Recurse -Force $BoxDir }
    New-Item -ItemType Directory -Force -Path $BoxDir | Out-Null
    Get-ChildItem $TempDir | Copy-Item -Destination $BoxDir

    [IO.File]::WriteAllText("$BoxDir\metadata.json", '{"provider":"virtualbox"}')
    [IO.File]::WriteAllText("$BoxDir\Vagrantfile",   'Vagrant.configure("2") { |c| }')

    Write-Host ""
    Write-Host "Listo. Caja '$BoxName' disponible en:"
    Write-Host "  $BoxDir"
    Write-Host ""
    Write-Host "Siguiente paso:"
    Write-Host "  vagrant up"

} catch {
    Write-Error "Fallo al crear la caja: $_"
} finally {
    vbm unregistervm $TempVm --delete 2>&1 | Out-Null
    Remove-Item -Recurse -Force $TempDir -ErrorAction SilentlyContinue
}
