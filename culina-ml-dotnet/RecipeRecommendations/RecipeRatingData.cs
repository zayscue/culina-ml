using Microsoft.ML.Data;

public class RecipeRating
{
    [LoadColumn(0)]
    public string User_id { get; set; }

    [LoadColumn(1)]
    public string Recipe_id { get; set; }

    [LoadColumn(2)]
    public float Rating { get; set; }
}

public class RecipeRatingPrediction
{
    public float Score { get; set; }
}