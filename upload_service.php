<?php
$target_dir = "logfiles/";
$target_file = $target_dir.basename($_FILES["fileToUpload"]["name"]);
$moved = move_uploaded_file($_FILES["fileToUpload"]["tmp_name"], $target_file);

if ($moved) {
echo "File ". htmlspecialchars( basename( $_FILES["fileToUpload"]["name"])). " has been uploaded successfully.";
} else {
echo "Error #".$_FILES["file"]["error"];}
?>
