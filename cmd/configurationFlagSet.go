package main

import (
	"flag"
	"fmt"
	"log"
	"os"
)

type queueConfig struct {
	Host string `yaml:Host`
	Port int `yaml:Port`
	Username string `yaml:Username`
	Password string `yaml:Password`
}

type blobStorageConfig struct {
	StorageType string "yaml:StorageType"
	Username string "yaml:Username"
	Password string "yaml:Password"
	Url string "yaml:Url"
}

type config struct {
	queueConfig
	blobStorageConfig
}

func createConfigurationFlagSet() (*flag.FlagSet, map[string]*string) {

	configFlagSet := flag.NewFlagSet("config", flag.ExitOnError)
	configFlagSet.Usage = func() {
		fmt.Println("Config is used to setup the configuration of required elements.")
	}
	kubeConfig := flag.String("kube_config",
		"",
		"Config that helps connect to the Cluster")

	subFlags := make(map[string]*string)
	subFlags["kube_config_path"] = kubeConfig

	return configFlagSet, subFlags

}

func handleConfigurationFlagSet(configFlagSet *flag.FlagSet, subFlags map[string]*string) (err error) {

	num_args := len(os.Args)
	if err = configFlagSet.Parse(os.Args[2:num_args-1]); err != nil {
		log.Println("Error occurred while parsing: ", err)
		return
	}

	//last_arg := os.Args[num_args]


	return
}

