package main

import (
	"flag"
	"fmt"
)


func createProjectFlagSet() *flag.FlagSet {

	projectFlagSet := flag.NewFlagSet("project", flag.ExitOnError)
	projectFlagSet.Usage = func() {
		fmt.Println("Manage Projects")
	}

	return projectFlagSet

}

func createExperimentFlagSet() *flag.FlagSet {

	projectFlagSet := flag.NewFlagSet("experiment", flag.ExitOnError)
	projectFlagSet.Usage = func() {
		fmt.Println("Manage Experiments for the current project")
	}

	return projectFlagSet

}

func createRunFlagSet() *flag.FlagSet {

	runExperimentFlagSet := flag.NewFlagSet("run", flag.ExitOnError)
	runExperimentFlagSet = func() {
		fmt.Println("Run an experiment")
	}
	runExperimentFlagSet.String("project_root")

}