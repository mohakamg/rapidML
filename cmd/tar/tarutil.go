package tar

import (
	"archive/tar"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

type Packager struct {
	// Source represents the path of the folder/file to tar or the
	// path to the tarball incase of untarring
	Source string
	// Destination Path is location of the output path
	DestinationPath string
	// This is an optional argument to rename the tarball.
	DestinationContentName string
}

func (packager *Packager) Tar() (err error) {

	// Make sure their exists a destination file/folder name
	if packager.DestinationContentName == "" {
		packager.DestinationContentName = filepath.Base(packager.Source)
	}

	// Create the destination tarfile path
	tarfilePath := filepath.Join(packager.DestinationPath,
		fmt.Sprintf("%s.tar", packager.DestinationContentName))

	// Create the tarfile
	tarfile, err := os.Create(tarfilePath)
	if err != nil { return err }
	defer tarfile.Close()

	// Create a writer to write to the file
	tarball := tar.NewWriter(tarfile)
	defer tarball.Close()

	err = filepath.Walk(packager.Source, func(path string, info os.FileInfo, err error) error {

		if err != nil { return err }

		// Create a file header
		header, err := tar.FileInfoHeader(info, info.Name())
		if err != nil { return err }

		// Build the complete path
		header.Name = path

		// Write the header to the tar ball
		if err = tarball.WriteHeader(header); err != nil { return err }

		// If the current path is a directory, we do not copy anything
		if info.IsDir() { return nil }

		// If the current path is not a directory but a file, open it
		file, err := os.Open(path)
		if err != nil { return err }
		defer file.Close()

		// And copy it into the tarball
		_, err = io.Copy(tarball, file)
		return err

	})

	return

}

func (packager *Packager) UnTar() (err error) {
	
	// Open the tarball
	reader, err := os.Open(packager.Source)
	if err != nil { return err }
	defer reader.Close()

	// Declare a reader
	tarReader := tar.NewReader(reader)

	// Loop over the contents of the tarball
	for {
		header, err := tarReader.Next()

		// If EOF file is reached quit
		if err == io.EOF { break } else if err != nil { return err }

		// Construct the path where to untar
		path := filepath.Join(packager.DestinationPath, header.Name)

		// Get the info of the current file/folder
		info := header.FileInfo()

		// If the current header in the loop says its a directory, create it and
		// move on to next file
		if info.IsDir() {
			if err = os.MkdirAll(path, info.Mode()); err != nil { return err }
			continue
		}

		// If the current header says its a file and not a folder
		file, err := os.OpenFile(path, os.O_CREATE | os.O_TRUNC | os.O_WRONLY, info.Mode())
		if err != nil { return err }
		defer file.Close()

		if _, err = io.Copy(file, tarReader); err != nil { return err }
	}
	
	return 
}
