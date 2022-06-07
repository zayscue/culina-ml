using Microsoft.ML;
using Microsoft.ML.Trainers;

MLContext mlContext = new MLContext();
(IDataView trainingDataView, IDataView testDataView) = LoadData(mlContext);
ITransformer model = BuildAndTrainModel(mlContext, trainingDataView);
EvaluateModel(mlContext, testDataView, model);
UseModelForSinglePrediction(mlContext, model);
SaveModel(mlContext, trainingDataView.Schema, model);

(IDataView training, IDataView test) LoadData(MLContext mlContext)
{
    var trainingDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "ml_dotnet_interactions_train.csv");
    var testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "ml_dotnet_interactions_train.csv");

    IDataView trainingDataView = mlContext.Data.LoadFromTextFile<RecipeRating>(trainingDataPath, hasHeader: true, separatorChar: ',');
    IDataView testDataView = mlContext.Data.LoadFromTextFile<RecipeRating>(testDataPath, hasHeader: true, separatorChar: ',');

    return (trainingDataView, testDataView);
}

ITransformer BuildAndTrainModel(MLContext mlContext, IDataView trainingDataView)
{
    IEstimator<ITransformer> estimator = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "UserIdEncoded", inputColumnName: "User_id")
        .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "RecipeIdEncoded", inputColumnName: "Recipe_id"));

    var options = new MatrixFactorizationTrainer.Options
    {
        MatrixColumnIndexColumnName = "UserIdEncoded",
        MatrixRowIndexColumnName = "RecipeIdEncoded",
        LabelColumnName = "Rating",
        NumberOfIterations = 10000,
        ApproximationRank = 100
    };

    var trainerEstimator = estimator.Append(mlContext.Recommendation().Trainers.MatrixFactorization(options));

    Console.WriteLine("=============== Training the model ===============");
    ITransformer model = trainerEstimator.Fit(trainingDataView);

    return model;
}

void EvaluateModel(MLContext mlContext, IDataView testDataView, ITransformer model)
{
    Console.WriteLine("=============== Evaluating the model ===============");
    var prediction = model.Transform(testDataView);

    var metrics = mlContext.Regression.Evaluate(prediction, labelColumnName: "Rating", scoreColumnName: "Score");

    Console.WriteLine("Root Mean Squared Error : " + metrics.RootMeanSquaredError.ToString());
    Console.WriteLine("RSquared: " + metrics.RSquared.ToString());
}

void UseModelForSinglePrediction(MLContext mlContext, ITransformer model)
{
    Console.WriteLine("=============== Making a prediction ===============");
    var predictionEngine = mlContext.Model.CreatePredictionEngine<RecipeRating, RecipeRatingPrediction>(model);

    var testInput = new RecipeRating { User_id = "auth0|6043f4dcb8458600693a374d", Recipe_id = "47dc8f77-42df-4667-929b-4835f2f32073" };

    var recipeRatingPrediction = predictionEngine.Predict(testInput);

    if (Math.Round(recipeRatingPrediction.Score, 1) > 3.5)
    {
        Console.WriteLine("Recipe " + testInput.Recipe_id + " is recommended for user " + testInput.User_id);
    }
    else
    {
        Console.WriteLine("Recipe " + testInput.Recipe_id + " is not recommended for user " + testInput.User_id);
    }

}

void SaveModel(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
{
    var modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "CollaborativeFilteringRecipeRecommendations.zip");

    Console.WriteLine("=============== Saving the model to a file ===============");
    mlContext.Model.Save(model, trainingDataViewSchema, modelPath);
}