all:
	python src/main.py

clean:
	rm -f charts/*

.PHONY: all clean
