# Crypto Prediction Project

## 🚨 **Important! Make sure you download the following files from the competition page:**

Before you start, **you MUST** download the required files from the [TopCoder competition page](https://www.topcoder.com/challenges/530f1495-c8f4-45df-b53e-a157c7e97d20?tab=details) to ensure the program runs correctly:

- **train.csv**
- **tester.jar**

> **⚠️ These files are critical! The program will NOT work without them! ⚠️**

---

## Prerequisites

This project requires the following files to be present in your project directory:

- `solution.py`
- `coin_decimals.csv`
- `README.md`
- `requirements.txt`

You must have **Python 3** and the **required dependencies** installed in your environment to run the program.

---

## Setting Up Your Environment

Follow these steps to set up the environment and run the program:

### Step 1: Clone the Repository

Clone the repository from GitHub:

```bash
git clone https://github.com/abdulrahmanRadan/crypto_prediction_project.git
cd crypto_prediction_project
```

### Step 2: Set Up a Virtual Environment

Set up a virtual environment to manage your project dependencies.

#### On Windows:

```bash
python -m venv venv
.\venv\Scripts\activate
```

#### On macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

Install the required Python libraries by running:

```bash
pip install -r requirements.txt
```

---

### Step 4: Prepare the Files

Before running the program, **make sure the following files** are in the **same directory** as the project:

- `train.csv`
- `tester.jar`

You can download these files from the competition page: [TopCoder Challenge](https://www.topcoder.com/challenges/530f1495-c8f4-45df-b53e-a157c7e97d20?tab=details).

---

### Step 5: Run the Program

After setting up the environment and ensuring the required files are in the same directory, run the following command to execute the program:

#### On **Windows**:

```bash
java -jar tester.jar -exec "python solution.py" -N 100 -data train.csv -debug
```

#### On **macOS/Linux**:

```bash
java -jar tester.jar -exec "python3 solution.py" -N 100 -data train.csv -debug
```

Make sure to use `python3` instead of `python` on macOS/Linux, as the default Python version may vary.

---

## Notes:

- **Make sure that both `train.csv` and `tester.jar` are present in your project directory.** The program will fail to execute without them.
- If you run into any issues or need further clarification, refer to the competition resources for assistance.

---

### **In Summary:**

1. **Download the required files** (`train.csv` and `tester.jar`) from the competition page.
2. **Set up your environment** and install dependencies.
3. **Run the provided command** to start the prediction process.
