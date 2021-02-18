package main

import (
	"fmt"
	"github.com/owulveryck/onnx-go"
	"github.com/owulveryck/onnx-go/backend/x/gorgonnx"
	"gorgonia.org/tensor"
	"log"

	//"gorgonia.org/gorgonia"
	"io/ioutil"
)

func main() {
	// Create a backend receiver
	backend := gorgonnx.NewGraph()
	// Create a model and set the execution backend
	model := onnx.NewModel(backend)

	// read the onnx model
	b, _ := ioutil.ReadFile("test_model.onnx")
	// Decode it into the model
	err := model.UnmarshalBinary(b)
	if err != nil {
		log.Fatalln(err)
	}

	input := tensor.New(tensor.WithShape(2, 1),
		tensor.WithBacking([]float32{5.00, 10}))
	err = model.SetInput(0, input)
	if err != nil {
		log.Fatalln(err)
	}
	err = backend.Run()
	if err != nil {
		log.Fatalln(err)
	}
	output, _ := model.GetOutputTensors()
	// write the first output to stdout
	fmt.Println(output)

}