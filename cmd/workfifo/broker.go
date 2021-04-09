// Package Work Fifo is a golang package for Asynchronous Messaging
// in a work queue format: https://www.rabbitmq.com/tutorials/tutorial-two-go.html
package workfifo

import (
	"bytes"
	"math/rand"
)

// Message Structure to store the contents of a message
type Message struct {
	ContentType string
	Headers map[string]string
	Body []byte
}

// Declare a Handler to handle the Message
type Handler func(message Message, ch chan []byte)

// Define what any broker needs to implement in
// order to be a part of this package.
type brokerAgent interface {
	Connect() error
	Disconnect() error
	Publish(queueName string, message Message) ([]byte, error)

}

func randInt(min int, max int) int {
	return min + rand.Intn(max-min)
}

func randomString(l int) string {
	bytes := make([]byte, l)
	for i := 0; i < l; i++ {
		bytes[i] = byte(randInt(65, 90))
	}
	return string(bytes)
}

func concatenateStrings(strings ...string) string {

	var buffer bytes.Buffer

	for _, str := range strings {
		buffer.WriteString(str)
	}

	return buffer.String()


}
