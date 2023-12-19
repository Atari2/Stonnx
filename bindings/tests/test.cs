using System;
using StonnxCLR;

public class HelloWorld {
    public static int Main() 
    {
        if (StonnxAPI.ONNXModel.Run("GPT2", StonnxAPI.Verbosity.Minimal, StonnxAPI.GraphFormat.None, StonnxAPI.ExecutionMode.FailFast)) {
            Console.WriteLine("Model execution succeeded");
            return 0;
        } else {
            Console.WriteLine("Model execution failed");
            Console.WriteLine("Error: " + StonnxAPI.LastError());
            return 1;
        }
    }
}