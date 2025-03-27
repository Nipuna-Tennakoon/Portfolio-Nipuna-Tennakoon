import streamlit as st
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql import Row

# Initialize Spark
spark = SparkSession.builder \
    .appName("Song Genre Prediction") \
    .config("spark.master", "local") \
    .getOrCreate()

label_values = ['pop', 'country', 'blues', 'jazz', 'reggae', 'rock', 'hip hop', 'Metal']  

# Load the trained model
model = PipelineModel.load("models/classifier_2")

# Define the Streamlit app
def main():
    # Set up the title and description
    st.title('I am Lyrics Classifier')
    st.write('Enter the lyrics of your song to check its genre.')

    # Add a text input for lyrics
    lyrics = st.text_input('Enter the lyrics:', '')

    # Add a submit button
    submitted = st.button('Submit')

    # Check if the submit button is clicked and lyrics are provided
    if submitted and lyrics:
        # Convert the lyrics to a DataFrame
        lyrics_df = spark.createDataFrame([Row(lyrics=lyrics)])

        # Make the prediction
        prediction = model.transform(lyrics_df)

        # Get the predicted genre
        predicted_genre = label_values[int(prediction.select("prediction").collect()[0][0])]

        # Get the probability scores for all genres
        probability_scores = prediction.select("probability").collect()[0][0]

        # Combine the genres and their corresponding probabilities
        genre_probabilities = {label_values[i]: probability_scores[i] for i in range(len(label_values))}

        # Show the prediction result
        st.write(f'Predicted Genre: {predicted_genre}')
        st.write('Probability Distribution:')
        st.bar_chart(genre_probabilities)


if __name__ == '__main__':
    main()
