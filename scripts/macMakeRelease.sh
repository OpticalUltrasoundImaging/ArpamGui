artifact_path=./build/clang-release/arpam_gui_qt/
artifact_name=arpam-gui-qt.app
archive_name="arpam-gui-mac-$(date +%Y%m%d).zip"

pushd $artifact_path
zip -r $archive_name $artifact_name
popd
mv $artifact_path/$archive_name .
