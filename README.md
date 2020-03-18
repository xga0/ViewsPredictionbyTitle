# ViewsPredictionbyTitle
Predict view of news/video by the title, using Keras deep learning library.

If you have "title-views" datasets for a news site or YouTube channel, you can try this model. This model performs a series of processing on the title. After Keras Tokenization and Embedding, the title will be input into a deep learning regression analysis model to predict the views.

Since this model performs Natural Logarithm (ln) processing on Y (that is, the views), a function that can predict the new title is provided at the end of the code.
