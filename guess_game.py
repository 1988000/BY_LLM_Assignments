# guess_game.py
import random
from pathlib import Path

def guess_game():
    print("Welcome to the Guess Game!")
    number = random.randint(1, 10)  # Random number between 1 and 10
    attempts = 0

    while True:
        try:
            guess = int(input("Guess a number between 1 and 10: "))
            attempts += 1
            if guess < number:
                print("Too low! Try again.")
            elif guess > number:
                print("Too high! Try again.")
            else:
                print(f"Congratulations! You guessed the number {number} in {attempts} attempts.")

                # Ensure folder exists
                save_folder = Path.home() / "BY_LLM_Assignments"
                save_folder.mkdir(parents=True, exist_ok=True)

                # Save the result
                file_path = save_folder / "guess_game_result.txt"
                with open(file_path, "w") as f:
                    f.write(f"Guessed number: {number}\nAttempts: {attempts}\n")
                print(f"Result saved to {file_path}")

                break
        except ValueError:
            print("Please enter a valid number.")

if __name__ == "__main__":
    guess_game()
