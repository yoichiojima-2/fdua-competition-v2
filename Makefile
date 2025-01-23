download:
	mkdir -p ./downloads
	gsutil -m cp -r gs://yo-personal/fdua/downloads/* ./downloads

upload:
	gsutil -m cp -r ./downloads gs://yo-personal/fdua/downloads/
