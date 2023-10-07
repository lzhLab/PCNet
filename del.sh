isort .
ls -lR | grep "/__pycache__" | awk -F ":" '{print "rm -rf " $1}' | sh
ls -ltR | grep "2023-01-66" | grep -v "drwxr" | awk -F ":" '{print "rm -rf " $1}' | sh
ls -lR| grep "^-" | wc -l
ls -lR | grep "^d" | wc -l