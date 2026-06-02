all:
	python src/main.py

clean:
	rm -f graphs/*

.PHONY: all clean
