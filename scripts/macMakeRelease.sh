artifact_path=./build/clang-release/ArpamGuiQt/
artifact_name=ArpamGuiQt.app
archive_name="ArpamGuiQt-mac-$(date +%Y%m%d).zip"

pushd $artifact_path
zip -r $archive_name $artifact_name
popd
mv $artifact_path/$archive_name .
