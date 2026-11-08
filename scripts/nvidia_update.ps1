param(
    [Parameter(Mandatory=$true)][ValidateSet('Find','Elevate','Install','Boot')][string]$Action,
    [Parameter(Mandatory=$true)][string]$Report,
    [Guid]$UpdateId,
    [int]$Revision
)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)

function Write-Result($Data) {
    $Data | ConvertTo-Json -Compress -Depth 5 | Set-Content -LiteralPath $Report -Encoding UTF8
}

function Get-BootStamp {
    return (Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToFileTimeUtc().ToString()
}

function Find-Driver {
    $devices = @(Get-PnpDevice -Class Display -PresentOnly |
        Where-Object InstanceId -Like 'PCI\VEN_10DE*')
    if (-not $devices.Count) { return @{ hardware = $false } }
    $ids = @($devices | ForEach-Object {
        foreach ($key in 'DEVPKEY_Device_HardwareIds','DEVPKEY_Device_CompatibleIds') {
            (Get-PnpDeviceProperty -InstanceId $_.InstanceId -KeyName $key -ErrorAction SilentlyContinue).Data
        }
    })
    if (-not $ids.Count) { return @{ hardware = $true } }
    $session = New-Object -ComObject Microsoft.Update.Session
    $session.ClientApplicationID = 'MTMC'
    $search = $session.CreateUpdateSearcher()
    $result = $search.Search("IsInstalled=0 and IsHidden=0 and Type='Driver'")
    if ($result.ResultCode -notin 2,3) { throw "Ошибка поиска Windows Update ($($result.ResultCode))." }
    $candidates = foreach ($update in $result.Updates) {
        if ($update.DriverClass -eq 'Display' -and $update.DriverManufacturer -match '^NVIDIA\b' -and
            $ids -contains $update.DriverHardwareID -and -not $update.InstallationBehavior.CanRequestUserInput) {
            $update
        }
    }
    $pick = $candidates | Sort-Object DriverVerDate -Descending | Select-Object -First 1
    if (-not $pick) { return @{ hardware = $true } }
    return @{ hardware = $true; id = $pick.Identity.UpdateID; revision = $pick.Identity.RevisionNumber;
              title = $pick.Title; hardware_id = $pick.DriverHardwareID }
}

try {
    if ($Action -eq 'Boot') {
        Write-Result @{ boot = Get-BootStamp }
    } elseif ($Action -eq 'Find') {
        try {
            Write-Result (Find-Driver)
        } catch {
            $present = @(Get-PnpDevice -Class Display -PresentOnly -ErrorAction SilentlyContinue |
                Where-Object InstanceId -Like 'PCI\VEN_10DE*').Count -gt 0
            Write-Result @{ hardware = $present; warning = $_.Exception.Message }
        }
    } elseif ($Action -eq 'Elevate') {
        Write-Output 'Ожидание подтверждения Windows и установки драйвера.'
        $arguments = '-NoProfile -ExecutionPolicy Bypass -File "{0}" -Action Install -Report "{1}" -UpdateId {2} -Revision {3}' -f $PSCommandPath, $Report, $UpdateId, $Revision
        $child = Start-Process -FilePath "$PSHOME\powershell.exe" -ArgumentList $arguments -Verb RunAs -Wait -PassThru
        exit $child.ExitCode
    } else {
        $offer = Find-Driver
        if ($offer.id -ne $UpdateId.ToString() -or $offer.revision -ne $Revision) {
            throw 'Предложение Windows Update изменилось. Повторите поиск драйвера.'
        }
        $session = New-Object -ComObject Microsoft.Update.Session
        $session.ClientApplicationID = 'MTMC'
        $search = $session.CreateUpdateSearcher()
        $result = $search.Search("IsInstalled=0 and Type='Driver' and UpdateID='$UpdateId' and RevisionNumber=$Revision")
        if ($result.Updates.Count -ne 1) { throw 'Выбранный драйвер больше недоступен.' }
        $update = $result.Updates.Item(0)
        $boot = Get-BootStamp
        if (-not $update.EulaAccepted) { $update.AcceptEula() }
        $list = New-Object -ComObject Microsoft.Update.UpdateColl
        [void]$list.Add($update)
        $download = $session.CreateUpdateDownloader()
        $download.Updates = $list
        if ($download.Download().ResultCode -ne 2 -or -not $update.IsDownloaded) {
            throw 'Windows Update не смог скачать драйвер.'
        }
        $installer = $session.CreateUpdateInstaller()
        $installer.Updates = $list
        $installed = $installer.Install()
        if ($installed.ResultCode -ne 2) { throw "Ошибка установки Windows Update ($($installed.ResultCode))." }
        Write-Result @{ reboot = [bool]$installed.RebootRequired; boot = $boot }
    }
} catch {
    Write-Result @{ error = $_.Exception.Message }
    exit 1
}
