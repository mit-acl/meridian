output_dir=$1
pushd $output_dir/..
gdown --folder "https://drive.google.com/drive/folders/1sL4NQRss1YiGHEEri9LXOg9N5yOGnOiQ?usp=sharing"
mv meridian_demo_data $output_dir
popd