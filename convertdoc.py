import codecs
sourceFileName = "./data/squad_nqg/test.json"
targetFileName = "./data/squad_nqg/testgood.json"
BLOCKSIZE = 1048576 # or some other, desired size in bytes
with codecs.open(sourceFileName, "r", "utf-8") as sourceFile:
    with codecs.open(targetFileName, "w", "utf-8") as targetFile:
        while True:
            contents = sourceFile.read(BLOCKSIZE)
            contents = contents.encode("ascii", "ignore")
            contents = contents.decode()
            if not contents:
                break
            targetFile.write(contents)