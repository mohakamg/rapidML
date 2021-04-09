package main

import (
	"flag"
	"log"
	"os"
	"fmt"
)

func main() {

	flag.Usage = func() {
		fmt.Printf("This CLI is used to run python jobs.")
	}

	// Create a flag to initialize the project
	initializeFlagSet := flag.NewFlagSet("init", flag.ExitOnError)
	initializeFlagSet.Usage = func() {
		fmt.Printf("This CLI Command is used to initialize a project.")
	}
	projectName := initializeFlagSet.String("project_name",
		"",
		"The name of the Project")
	projectRoot := initializeFlagSet.String("project_root",
		"",
		"By default the project root is considered from wherever the project is executed unless specified here")

	// Create a Projects flag
	projectsFlagSet := flag.NewFlagSet("project", flag.ExitOnError)
	projectsFlagSet.Usage = func() {
		fmt.Printf("This CLI Command is used to initialize a project.")
	}

	// Create a flag set for run. This is used to
	// run a python program
	runFlagSet := flag.NewFlagSet("run", flag.ExitOnError)

	// Describe the inputs
	_ = runFlagSet.String("python_file",
		"",
		"The file that needs to be executed")

	switch os.Args[1] {
	case "init":
		HandleInit(initializeFlagSet, projectName, projectRoot)
	case "run":
		fmt.Println("Run")
	default:
		fmt.Println("Unknown command")
	}

}

func HandleInit(initCmd *flag.FlagSet, project_name *string, project_root *string) {

	if err := initCmd.Parse(os.Args[2:]); err != nil {
		log.Println("Error occurred while parsing: ", err)
	}

	if *project_name == "" {
		log.Println("Cannot initialize project without a Project name being specified.")
		initCmd.PrintDefaults()
		os.Exit(1)
	}

	if *project_root == "" {
		if currDir, err := os.Getwd(); err != nil {
			log.Println("Could not fetch working dir: ", err)
		} else {
			log.Println("Using current directory: ", currDir)
		}
	}

}