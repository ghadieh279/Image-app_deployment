import streamlit as st
import tensorflow as tf
from PIL import Image, ExifTags
import uuid
import rawpy
import io
import os
#import wxPython
import time 

st.set_page_config(layout="wide")

@st.cache_resource(show_spinner=False)
def load_model():
    with st.spinner("Loading the image classification APP..."):
        model = tf.keras.models.load_model('https://github.com/ghadieh279/Image-app_deployment/Tree/main/img_saved_model.h5')
    return model

model = load_model()
IMAGE_SIZE = (180, 180)
CLASS_NAMES = ["airport", "animal", "city", "food", "human", "nature"]


def predict_image_class_in_memory(image_bytes):
    img = tf.keras.preprocessing.image.load_img(io.BytesIO(image_bytes), target_size=IMAGE_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    top_k = tf.math.top_k(predictions, k=2) # Get the top 2 predictions
    first_pred_class = CLASS_NAMES[top_k.indices[0][0]]
    first_pred_accuracy = top_k.values[0][0] * 100
    second_pred_class = CLASS_NAMES[top_k.indices[0][1]]
    second_pred_accuracy = top_k.values[0][1] * 100
    return (first_pred_class, first_pred_accuracy), (second_pred_class, second_pred_accuracy)

def read_image_in_memory(image_bytes, filename):
    _, ext = os.path.splitext(filename)
    if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.cr2']:
        if ext.lower() == '.cr2':
            with rawpy.imread(io.BytesIO(image_bytes)) as raw:
                rgb = raw.postprocess()
            image = Image.fromarray(rgb)
        else:
            image = Image.open(io.BytesIO(image_bytes))
        return image
    else:
        return None

def correct_image_orientation(image):
    try:
        exif = image._getexif()
        if exif is not None:
            exif = dict(exif.items())
            orientation_key = next((key for key, val in ExifTags.TAGS.items() if val == 'Orientation'), None)
            if orientation_key is not None and orientation_key in exif:
                orientation = exif[orientation_key]
                if orientation == 3:
                    image = image.rotate(180, expand=True)
                elif orientation == 6:
                    image = image.rotate(270, expand=True)
                elif orientation == 8:
                    image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        pass
    return image

def select_folder():
    app = wx.App(False)
    dialog = wx.DirDialog(None, "Choose a folder", style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
    if dialog.ShowModal() == wx.ID_OK:
        folder_path = dialog.GetPath()
    else:
        folder_path = None
    dialog.Destroy()
    app.Destroy()
    return folder_path


def move_images(image_bytes_list, initial_predictions, user_selected_predictions, root_folder, use_combined_folder=False):
    for img_bytes, initial_pred, user_selected_class in zip(image_bytes_list, initial_predictions, user_selected_predictions):
        # Unpack the initial predictions tuple
        first_pred_class, first_pred_accuracy, second_pred_class, second_pred_accuracy = initial_pred

        # Determine the folder path based on the prediction accuracy difference
        if use_combined_folder and st.session_state['ok'] == True and abs(first_pred_accuracy - second_pred_accuracy) <= st.session_state['avg']:
            predicted_folder_path = os.path.join(root_folder, f"{first_pred_class}_and_{second_pred_class}")
        else:
            predicted_folder_path = os.path.join(root_folder, user_selected_class)

        # Create the folder if it doesn't exist
        if not os.path.exists(predicted_folder_path):
            os.makedirs(predicted_folder_path)

        # Save the image in the appropriate folder
        image_output_path = os.path.join(predicted_folder_path, str(uuid.uuid4()) + '.jpg')
        with open(image_output_path, "wb") as f:
            f.write(img_bytes)


def width(size_factor): 
    if size_factor == 3 :
        image_width = 390
        
    elif size_factor == 4 :
        image_width = 295  
    
    elif size_factor == 5 :
        image_width = 235
    elif size_factor == 6 :
        image_width = 195
        
    else:
        image_width = 165
    return image_width




warning = """
“If multiple images are uploaded with the same name, only the first will be considered. Rename any additional images to ensure all are uploaded.”
"""

def stream_data():
    for word in warning.split(" "):
        yield word + " "
        time.sleep(0.2)


warning2 = """“Please upload the image files, the app will predict their classes once you upload them.”"""

def stream_data2():
    for word in warning2.split(" "):
        yield word + " "
        time.sleep(0.2)


def reset_app():
    # Save the current script path
    script_path = os.path.abspath(__file__)

    if st.info("The app will restart in a new tab."):
        time.sleep(3)
              
        # Restart the app by running the script again
        os.system(f"streamlit run {script_path}")
     
# Main function of the app
def main():
    st.title('Image Classification App')
    tab1, tab2 = st.tabs(['Upload', 'Predictions'])

    # Initialize session state variables
    if 'uploaded_files' not in st.session_state:
        st.session_state['uploaded_files'] = {}
    if 'predictions' not in st.session_state:
        st.session_state['predictions'] = []
    if 'file_names' not in st.session_state:
        st.session_state['file_names'] = set()
    if 'non_image_files' not in st.session_state:
        st.session_state['non_image_files'] = []
    if 'initial_predictions' not in st.session_state:
        st.session_state['initial_predictions'] = []
    if 'user_selected_predictions' not in st.session_state:
        st.session_state['user_selected_predictions'] = []
    if 'processed_images' not in st.session_state:
        st.session_state['processed_images'] = {}

    st.session_state['error'] = False
    st.session_state['ok'] = False
    images_uploaded = False
    
    with tab1:
        
        with st.popover("Warning"):
            st.warning("“Ensure unique names for each image upload.”")
            if st.toggle("More explanation"):
                st.write_stream(stream_data)

        uploaded_files = st.file_uploader("Drag and Drop Images Here", type=None, accept_multiple_files=True, key='file_uploader')

    if uploaded_files:
        new_files = [file for file in uploaded_files if file.name not in st.session_state['file_names']]
        total_files = len(new_files)
        images_uploaded = True
    
        # Create a container for the progress bar and text
        with st.container():
            pt = st.empty()
            c1, c2 = st.columns([4,1])  # Adjust the ratio as needed
            my_bar = c1.progress(0)  # Initialize the progress bar in the first column
            progress_text2 = c2.empty()  # Placeholder for the text in the second column
    
        for index, uploaded_file in enumerate(new_files):
            remaining_images = total_files - (index + 1)  # Calculate remaining images
            progress_percentage = int((index + 1) / total_files * 100)
            progress_text = f"Your patience is appreciated; predictions coming up soon!... ({remaining_images} remaining)"
            
            # Update the progress bar and text
            my_bar.progress(progress_percentage)
            progress_text2.text(f"{index + 1}/{total_files}")  # Update the text in the second column

            # Display the final message below the progress bar after the loop
            pt.text(progress_text)
            
            st.session_state['file_names'].add(uploaded_file.name)
            if uploaded_file.name not in st.session_state['processed_images']:
                bytes_data = uploaded_file.read()
                image = read_image_in_memory(bytes_data, uploaded_file.name)



                if image is not None:
                    image = correct_image_orientation(image)
                    buffered = io.BytesIO()
                    image.save(buffered, format="JPEG")
                    img_bytes = buffered.getvalue()
    
                    # Save the processed image bytes in session state
                    st.session_state['processed_images'][uploaded_file.name] = img_bytes
    
                    # Perform prediction and store the results in session state
                    if uploaded_file.name not in st.session_state['predictions']:
                        (first_pred_class, first_pred_accuracy), (second_pred_class, second_pred_accuracy) = predict_image_class_in_memory(img_bytes)
                        st.session_state['predictions'].append((uploaded_file.name, first_pred_class, first_pred_accuracy, second_pred_class, second_pred_accuracy))
                else:
                    # Handle non-image files
                    if uploaded_file.name not in st.session_state['non_image_files']:
                        st.session_state['non_image_files'].append(uploaded_file.name)


        my_bar.empty()
        pt.empty()
        progress_text2.empty()


    else:
        war = st.write_stream(stream_data2)    

    with tab1:        
        if st.session_state['non_image_files']:
            st.session_state['error'] = True
            c1 ,c2 = st.columns(2)
            with c1 :
                error_message = st.error(f"The following files are not supported image formats: {', '.join(st.session_state['non_image_files'])}")
            with c2:
                button_placeholder = st.empty()
                if button_placeholder.button("OK", key="error_ok"):
                    error_message.empty()
                    st.session_state['non_image_files'].clear()
                    button_placeholder.empty()
                    st.session_state['error'] = False

    if uploaded_files:
        
        st.markdown("---")

        st.markdown(">>>")

        if st.button('Reset App'):
            reset_app()


    with tab2:
        if images_uploaded:
            if not st.session_state['error'] and st.session_state['predictions']:
                size_factor = st.slider('Adjust Image Size and Columns', 3, 7, 5, key='size_slider')
                columns = max(3, size_factor)
                image_width = width(size_factor)
                cols = st.columns(columns)
                
                with st.spinner("Loading images..."): 
                    for index, (filename, first_pred_class, first_pred_accuracy, second_pred_class, second_pred_accuracy) in enumerate(st.session_state['predictions']):
                        img_bytes = st.session_state['processed_images'][filename]
                        if len(st.session_state['initial_predictions']) <= index:
                            st.session_state['initial_predictions'].append((first_pred_class, first_pred_accuracy, second_pred_class, second_pred_accuracy))
                

                    for index, (filename, first_pred_class, first_pred_accuracy, second_pred_class, second_pred_accuracy) in enumerate(st.session_state['predictions']):
                        img_bytes = st.session_state['processed_images'][filename]
                        with cols[index % columns]:
                            image = Image.open(io.BytesIO(img_bytes))
                            with st.container():
                                st.image(image, width=image_width)
                                text_width = int(image_width * columns - 2 / (index % columns + 1))
                                st.write(f"Pred 1: **{st.session_state['initial_predictions'][index][0]}** - Acc: **{st.session_state['initial_predictions'][index][1]:.2f}%**", width=text_width)
                                
                                # Display the second prediction and allow the user to select a new prediction class
                                st.write(f"Pred 2: **{second_pred_class}** - Acc: **{second_pred_accuracy:.2f}%**", width=text_width)
                                new_pred_class = st.selectbox("Choose Prediction:", [first_pred_class, second_pred_class], key=f"selectbox_{index}")
    
                            # Update the user's selected predictions list with the chosen class
                            if len(st.session_state['user_selected_predictions']) <= index:
                                st.session_state['user_selected_predictions'].append(new_pred_class)
                            else:
                                st.session_state['user_selected_predictions'][index] = new_pred_class


 
                           
                            if 'space' not in st.session_state:
                                st.session_state['space'] = []
    
    
            
            elif st.session_state['error'] == True:
                st.warning("Please return to 'Upload' tab to resolve the issues before viewing predictions.")   
                st.write(" ", height=100)
                st.write(" ", height=100)
                st.write(" ", height=100)
                st.write(" ", height=100)
                st.write(" ", height=100)        
                st.write(" ", height=100)
                st.write(" ", height=100)
                st.write(" ", height=100)
                st.write(" ", height=100)
                st.write(" ", height=100)
                st.write(" ", height=100)
                st.write(" ", height=100)
                st.write(" ", height=100)
                st.write(" ", height=100)
                st.write(" ", height=100)        
                st.write(" ", height=100)
                st.write(" ", height=100)
                st.write(" ", height=100)
                st.write(" ", height=100)
                st.write(" ", height=100)  
    
    
            if 'space' not in st.session_state:
                st.write(" ", height=100)
                st.write(" ", height=100)
                st.write(" ", height=100)
                st.write(" ", height=100)
                st.write(" ", height=100)        
                st.write(" ", height=100)
                st.write(" ", height=100)
                st.write(" ", height=100)
                st.write(" ", height=100)
                st.write(" ", height=100)
                st.write(" ", height=100)
                st.write(" ", height=100)
                st.write(" ", height=100)
                st.write(" ", height=100)
                st.write(" ", height=100)        
                st.write(" ", height=100)
                st.write(" ", height=100)
                st.write(" ", height=100)
                st.write(" ", height=100)
                st.write(" ", height=100)
   
            if images_uploaded and st.session_state['error'] == False:
                st.markdown("---")
                use_combined_folder = st.checkbox("Use combined folder for close predictions")

                colm1, colm2 = st.columns([1, 3])
                if use_combined_folder:
                    with colm1 :
                        st.session_state['avg'] = st.number_input("Minimum accuracy difference for combined folder (default: 10)", min_value=0.0, max_value=100.0, value=10.0, step=5.0)

                
                st.session_state['Browse'] = False
                
                clm1, clm2 = st.columns([4, 1])
                clm3, clm4 = st.columns([4, 1])
                clm5, clm6 = st.columns([4, 1])
                with clm1 :
                    st.info("Hit ‘Browse’  – find the perfect spot for your pics!” ")
                with clm2:
                    if st.button("Browse"):
                        st.session_state['Browse'] = True
                #with clm3:
                    #if st.session_state['Browse'] == True:
                        #folder_path = select_folder()
                        #if folder_path:
                            #st.session_state['selected_folder'] = folder_path
                            #st.info(f'You selected this folder path: {folder_path}.')
                            #with clm5: 
                            #    st.info('Press "Submit" to save the images according to predictions or "Browse" to choose a different folder.')
                with clm6: 
                    button_submit = st.empty()
                    
                    if 'selected_folder' in st.session_state :
                        if st.session_state['selected_folder'] != None:
                            if button_submit.button("submit", key="button_submit"):
                                folder_path = st.session_state['selected_folder']
                                pth = folder_path.split('\\')[-1]
                                if not os.path.exists(folder_path):
                                    os.makedirs(folder_path)
                                image_bytes_list = [st.session_state['processed_images'][filename] for index, (filename, _, _, _, _) in enumerate(st.session_state['predictions'])]
                                initial_predictions = st.session_state['initial_predictions']
                                user_selected_predictions = st.session_state['user_selected_predictions']
                            
                                move_images(image_bytes_list, initial_predictions, user_selected_predictions, folder_path, use_combined_folder)
                
                                st.toast(f"Images successfully saved in {pth} folder")
                                st.balloons()
                                # Clear the predictions after moving the images
                                button_submit.empty()
                                st.session_state['predictions'] = []
                                st.session_state['initial_predictions'] = []
                                st.session_state['user_selected_predictions'] = []
                                st.session_state['selected_folder'] = None



if __name__ == '__main__':
    main()
