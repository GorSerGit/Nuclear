# Совместимость со старым запуском. Установка выполняется на вкладке «Зависимости».
param([string]$ParseOnly = "")

function Get-TorchIndex([string]$smi) {
    # Какая сборка torch подходит драйверу. nvidia-smi пишет "CUDA Version: 12.8"
    if ($smi -notmatch 'CUDA (?:UMD )?Version:\s*(\d+\.\d+)') { return $null }
    $v = [version]$Matches[1]
    if ($v -ge [version]'12.8') { return 'cu128' }
    if ($v -ge [version]'12.6') { return 'cu126' }
    if ($v -ge [version]'11.8') { return 'cu118' }
    return $null
}

if ($ParseOnly) { Get-TorchIndex $ParseOnly; exit 0 }

& (Join-Path $PSScriptRoot '!start.bat') --dependencies
exit $LASTEXITCODE
