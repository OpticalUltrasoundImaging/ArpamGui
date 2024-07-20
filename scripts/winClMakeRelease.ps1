$archivePath = "ArpamGuiQt-Win64-$(Get-Date -format 'yyyyMMdd').zip"

$compress = @{
    Path = ".\build\cl\ArpamGuiQt\Release\*"
    CompressionLevel = "Optimal"
    DestinationPath = $archivePath
}
Compress-Archive @compress -Update
