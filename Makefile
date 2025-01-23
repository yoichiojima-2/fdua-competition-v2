download:
	mkdir -p ./downloads
	gsutil -m cp -r gs://yo-personal/fdua/downloads/* ./downloads

upload:
	gsutil -m cp -r ./downloads gs://yo-personal/fdua/

unzip:
	cd downloads && \
	find . -name "*.zip" -print -exec unzip {} \;

clean:
	find . -type d -name "*__MACOS*" -print -exec rm -r {} +
	find . -type f -name ".DS_Store" -print -exec rm -r {} +
