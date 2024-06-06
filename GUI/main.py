from tkinter import *
from ttkbootstrap.constants import *
from PIL import ImageTk, Image
import ttkbootstrap as tb
from tkinter.ttk import Combobox
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

root = tb.Window(themename ="superhero")
root.geometry('500x550')
root.minsize(500, 500)
root.iconbitmap('bdu_logo.ico')
default_font = ('Karla', 11)

nanoparticle_frame = None  # Define nanoparticle_frame as a global variable

def reset_nanoparticle_page():
    if nanoparticle_frame:
        nanoparticle_frame.pack_forget()
        create_nanoparticle_page()  # Re-create nanoparticle page to show placeholder image
        
def replace_root_with_nanoparticle():
    root.title('Nanoparticle Size')
    root_frame.pack_forget()
    message_label.config(text="Please enter the following synthesis parameters values:")
    message_label.pack(pady=20)
    # Create the Nanoparticle Size page if it's not created yet
    if not nanoparticle_frame:
        create_nanoparticle_page()
    # Show the Nanoparticle Size page
    nanoparticle_frame.pack(fill=BOTH, expand=True)
    
def create_nanoparticle_page():
    global nanoparticle_frame, band_gap_spinbox, reaction_slider, calcination_slider, reaction_hr_spinbox, calcination_hr_spinbox, selected_ph, selected_precursor, selected_method, concentration_spinbox
    nanoparticle_frame = Frame(root)
    nanoparticle_frame.pack(fill=BOTH, expand=True)
    
    input_frame = Frame(nanoparticle_frame)
    input_frame.pack(fill=X, padx=10, pady=10)
    
    # Create a label for reaction hour
    Label(input_frame, text="Energy band gap ( eV ) : ", font=default_font).grid(row=0, column=0, sticky=W)

    # Create a Spinbox for reaction hour input
    band_gap_spinbox = Spinbox(input_frame, from_=3.0, to=3.90, increment=0.01, font=default_font)
    band_gap_spinbox.grid(row=0, column=1, sticky=W)
    
    # Function to update the reaction temperature label
    def reaction_temperature(value):
        reaction_label.config(text=f"{value} \u00b0C")

    # Create a label for reaction temperature
    Label(input_frame, text="Reaction temperature ( \u00b0C ) :", font=default_font).grid(row=1, column=0, sticky=W)

    # Create a Scale for temperature input
    reaction_slider = Scale(input_frame, from_=25, to=180, resolution=5, length=205, orient='horizontal', font=default_font, command=reaction_temperature)
    reaction_slider.grid(row=1, column=1, sticky=W)

    # Create a label to display the selected temperature
    reaction_label = Label(input_frame,font=default_font)
    reaction_label.grid(row=1, column=2, sticky=W)
    
    # Function to update the calcination temperature label
    def calcination_temperature(value):
       calcination_label.config(text=f"{value} \u00b0C")

    # Create a label for calcination temperature
    Label(input_frame, text="Calcination temperature ( \u00b0C ) : ", font=default_font).grid(row=2, column=0, sticky=W)

    # Create a Scale for calcination temperature input
    calcination_slider = Scale(input_frame, from_=70, to=400, resolution=25, length=205, orient='horizontal', font=default_font, command=calcination_temperature)
    calcination_slider.grid(row=2, column=1, sticky=W)

    # Create a label to display the selected temperature
    calcination_label = Label(input_frame,font=default_font)
    calcination_label.grid(row=2, column=2, sticky=W)
    
    # Create a label for reaction hour
    Label(input_frame, text="Reaction hour ( hr ) : ", font=default_font).grid(row=3, column=0, sticky=W)

    # Create a Spinbox for reaction hour input
    reaction_hr_spinbox = Spinbox(input_frame, from_=0.5, to=24, increment=0.5, font=default_font)
    reaction_hr_spinbox.grid(row=3, column=1, sticky=W)
    
    # Create a label for calcination hour
    Label(input_frame, text="Calcination hour ( hr ) : ", font=default_font).grid(row=4, column=0, sticky=W)

    # Create a Spinbox for calcination hour input
    calcination_hr_spinbox = Spinbox(input_frame, from_=1, to=24, increment=1, font=default_font)
    calcination_hr_spinbox.grid(row=4, column=1, sticky=W)
    
    # Create a label for pH
    Label(input_frame, text="pH :", font=default_font).grid(row=5, column=0, sticky=W)
    
    
    # Create a label for Synthesis Method
    Label(input_frame, text="Synthesis method :", font=default_font).grid(row=7, column=0, sticky=W)

    # Create a Combobox for synthesis method selection
    method_options = ["Sol-gel", "Green", "Hydrothermal", "Solvothermal"]
    selected_method = Combobox(input_frame, values=method_options, font=default_font)
    selected_method.current(0)  # Set the default value
    selected_method.grid(row=7, column=1, sticky=W)
    
    # Create a Combobox for precursor selection
    precursor_options = ["Zinc acetate", "Zinc nitrate"]
    selected_precursor = Combobox(input_frame, values=precursor_options, font=default_font)
    selected_precursor.current(0)  # Set the default value
    selected_precursor.grid(row=6, column=1, sticky=W)
    
    # Define pH options
    ph_options = list(range(7, 15))

    # Create a Combobox for pH selection
    selected_ph = Combobox(input_frame, values=ph_options, font=default_font)
    selected_ph.current(0)  # Set the default value
    selected_ph.grid(row=5, column=1, sticky=W)

    # Create a label for Precursor
    Label(input_frame, text="Precursor :", font=default_font).grid(row=6, column=0, sticky=W)

    # Create a label for concentration
    Label(input_frame, text="Concentration ( M ) : ", font=default_font).grid(row=8, column=0, sticky=W)

    # Create a Spinbox for concentration input
    concentration_spinbox = Spinbox(input_frame, from_=0.01, to=2, increment=0.01, font=default_font)
    concentration_spinbox.grid(row=8, column=1, sticky=W)

    estimate_button = Button(input_frame, text="Estimate", command=perform_estimation, font=default_font)
    estimate_button.grid(row=9, columnspan=2, pady=10)
    reset_button = Button(input_frame, text="Reset", command=reset_nanoparticle_page, font=default_font)
    reset_button.grid(row=9, column=2, pady=10)
    
    # Create placeholder image
    global placeholder_image
    placeholder_image = PhotoImage(file="machine-learning.png")
    global image_label
    image_label = Label(nanoparticle_frame, image=placeholder_image)
    image_label.image = placeholder_image  # Keep reference to the image to avoid garbage collection
    image_label.pack(pady=10)
    
    global prediction_label
    prediction_label = Label(nanoparticle_frame, text="", font=("Karla", 12))
    prediction_label.pack(pady=5)
    
    global prediction_text
    prediction_text = StringVar()
    prediction_text.set("")  # Set default prediction text
    
    global prediction_result_label
    prediction_result_label = Label(nanoparticle_frame, textvariable=prediction_text, font=("Karla", 25, "bold"))
    prediction_result_label.pack(pady=10)
    
def perform_estimation():

    band_gap_value = band_gap_spinbox.get()
    reaction_temp_value = reaction_slider.get()
    calcination_temp_value = calcination_slider.get()
    reaction_hour_value = reaction_hr_spinbox.get()
    calcination_hour_value = calcination_hr_spinbox.get()
    pH_value = selected_ph.get()
    precursor_value = selected_precursor.get()
    synthesis_method_value = selected_method.get()
    precursor_concentration_value = concentration_spinbox.get()
    
    # Setting SEED for reproducibility
    SEED = 42

    # Importing dataset
    ZnO_data_set = pd.read_csv('Original_ZnO_dataset_2.csv')

    # Create a separate LabelEncoder object for each categorical column
    le_synthesis_method = LabelEncoder()
    le_precursor = LabelEncoder()

    # Fit the label encoder and transform each categorical column individually
    ZnO_data_set["Synthesis method"] = le_synthesis_method.fit_transform(ZnO_data_set["Synthesis method"])
    ZnO_data_set["Precursor"] = le_precursor.fit_transform(ZnO_data_set["Precursor"])

    # Handling missing values by replacing them with the mean of each feature
    X = ZnO_data_set.iloc[:, :-1]
    X.fillna(X.mean(), inplace=True)  # Replace missing values with the mean of each feature

    y = ZnO_data_set.iloc[:, -1]
    y.fillna(y.mean(), inplace=True)  # Replace missing values with the mean of the target variable

    # Splitting dataset
    train_X, test_X, train_y, test_y = train_test_split(X, y, 
                                                    test_size = 0.25, 
                                                    random_state = SEED)

    # Instantiate XGBoost Regressor
    xgb_regressor = XGBRegressor(
                            learning_rate=0.022, # [0.01, 0.3]
                            n_estimators=295, # [100, inf]
                            max_depth=12, # [3, 10]
                            random_state=SEED,
                            min_child_weight=0.3,
                            subsample=0.5, # (0.0, 1.0]
                            colsample_bytree = 1.0, # [0.5, 1]
                            gamma=0.5 # [0.0, 0.5]
                            )
 
    # Fit to training set
    xgb_regressor.fit(train_X, train_y)
    
    # Initialize LabelEncoders for categorical variables
    le_synthesis_method = LabelEncoder()
    le_precursor = LabelEncoder()

    # Fit the LabelEncoders with possible categories
    possible_synthesis_methods = ['Sol-gel', 'Green', 'Solvothermal', 'Hydrothermal']
    possible_precursors = ['Zinc nitrate', 'Zinc acetate']

    le_synthesis_method.fit(possible_synthesis_methods)
    le_precursor.fit(possible_precursors)

    # Create a dictionary with the provided values
    data = {
    'Energy band gap (ev)': [float(band_gap_value)],
    'Reaction temperature (C)': [float(reaction_temp_value)],
    'Calcination temperature (C)': [float(calcination_temp_value)],
    'Reaction hour (hr)': [float(reaction_hour_value)],
    'Calcination hour (hr)': [float(calcination_hour_value)],
    'Synthesis method': [synthesis_method_value],
    'Precursor': [precursor_value],
    'pH': [float(pH_value)],
    'Precursor concentration (M)': [float(precursor_concentration_value)]
    }

    # Create the DataFrame
    new_data = pd.DataFrame(data)

    # Handle missing values by replacing them with the mean of each feature
    new_data.fillna(X.mean(), inplace=True)

    # If there are categorical variables, encode them
    new_data["Synthesis method"] = le_synthesis_method.transform(new_data["Synthesis method"])
    new_data["Precursor"] = le_precursor.transform(new_data["Precursor"])

    # Predict on new data
    new_predictions = xgb_regressor.predict(new_data)
    
    prediction_value = np.round(new_predictions, 2) # Sample value
    
    # Update the prediction text
    prediction_text.set(prediction_value)
    
    # Remove the placeholder image
    image_label.pack_forget()

    # Update the prediction label
    prediction_label.config(text="The approximate size of the nanoparticle is: ",)
    
    
def return_to_home():
    root.title('Home')
    root.iconbitmap('bdu_logo.ico')
    root_frame.pack(side=TOP, fill=X)
    message_label.config(text=("Before using the program, we recommend reading the About section. This will provide you with important information about the app and help you use it effectively."))
    message_label.pack(pady=20)
    nanoparticle_btn.pack(pady=10)
    reset_nanoparticle_page() 
    if nanoparticle_frame:
        nanoparticle_frame.pack_forget() 

def open_about():
    root.title('About')
    root.iconbitmap('bdu_logo.ico')
    root_frame.pack_forget()
    message_label.config(text="Hi, I am the About window")
    message_label.pack(pady=20)
    reset_nanoparticle_page() 
    if nanoparticle_frame:
        nanoparticle_frame.pack_forget()   # Reset nanoparticle page when opening about window

def create_home_page():
    global message_label, root_frame, nanoparticle_btn
    root.title('Home')
    root.iconbitmap('bdu_logo.ico')

    message_text = ("Before using the program, we recommend reading the About section. This will provide you with important information about the app and help you use it effectively.")
    message_label = Label(root, text=message_text, wraplength=380, justify='center', font=default_font)

    root_frame = Frame(root)

    nanoparticle_btn = Button(root_frame, text='Nanoparticle Size', command=replace_root_with_nanoparticle, font=default_font)
    nanoparticle_btn.pack(pady=10)
    
    return_to_home()

# Create a menu bar
menu_font = ("Karla", 18, "bold")

menubar = Menu(root)
root.config(menu=menubar)

# Add 'Home' and 'About' to the menu bar
home_menu = Menu(menubar, tearoff=0, font=menu_font)
menubar.add_cascade(label='Home', command=return_to_home)

about_menu = Menu(menubar, tearoff=0,  font=menu_font)
menubar.add_cascade(label='About', command=open_about)

create_home_page()

root.mainloop()
