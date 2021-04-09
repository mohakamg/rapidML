package main

import (
	"../../cmd/pyrunner"
	"fmt"
)

func main() {

	packager := pyrunner.Packager{
		Source:                 "./cmd.tar",
		DestinationPath:        "./cmduntar",
		//DestinationContentName: "cmd",
	}

	if err := packager.UnTar(); err != nil {
		fmt.Println(err)
	}

}
