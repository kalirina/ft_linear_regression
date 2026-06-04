import json
from plot import point_in_graph


def predict(x):
    try:
        with open("data/thetas.json", "r") as read_file:
            data = json.load(read_file)
            return data["theta0"] + data["theta1"] * x
    except:
        return 0


def main():
    try:
        mileage = float(input("Enter mileage: "))
    except (ValueError, EOFError):
        print("Error: invalid or missing input.")
        return
    except KeyboardInterrupt:
        print("\nProgram cancelled.")
        return

    if mileage < 0:
        print("Error: mileage cannot be negative.")
        return
    if mileage > 240000:
        print("Warning: mileage exceeds training data range.")
        return

    prediction = predict(mileage)
    print("Estimated price: ", prediction)
    if prediction:
        point_in_graph(mileage, prediction)


if __name__ == "__main__":
    main()
