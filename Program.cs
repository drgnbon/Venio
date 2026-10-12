


using System.Threading.Tasks.Dataflow;
using Venio.Venio.ActivationFunctions;
using Venio.Venio.LossFunctions;
using Venio.Venio.Model;
using Venio.Venio.Optimizers;
using static Venio.Venio.Model.Model;

public class Program
{
    public static void Main(string[] args)
    {

        //Activation function ++ 
        //Layer ++
        //Loss function
        //Model
        //convention
        //optimizer


        Model model = ModelBuilder
            .Create()
            .InputLayer(20)
            .SequentialLayer(10,new Logistic())
            .SequentialLayer(5,new Logistic())
            .LossFunction(LossType.Square)
            .Build();


        
    }
}


