artifact_path=./build/clang/ArpamGuiQt/Release
artifact_name=ArpamGuiQt.dmg
archive_name="ArpamGuiQt-mac-$(date +%Y%m%d).dmg"

cp -r $artifact_path/$artifact_name $archive_name
