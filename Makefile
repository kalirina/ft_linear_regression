.venv/bin/activate:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

train: .venv/bin/activate
	.venv/bin/python src/train.py

predict: .venv/bin/activate
	.venv/bin/python src/predict.py

clean:
	rm -f graphs/*
	rm -f data/thetas.json

fclean: clean
	rm -rf .venv

.PHONY: train predict clean fclean
