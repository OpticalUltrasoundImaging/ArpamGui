$archivePath = "ArpamGuiQt-$(Get-Date -format 'yyyyMMdd').zip"

$compress = @{
    Path = ".\build\win64\arpam_gui_qt\Release\*"
    CompressionLevel = "Optimal"
    DestinationPath = $archivePath
}
Compress-Archive @compress -Update
