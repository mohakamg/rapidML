package main
import (
	"../../cmd/workfifo"
	"encoding/json"
	"log"
	"strconv"
	"time"
)

func PrintTable(message workfifo.Message, ch chan []byte) {
	type TableMsg struct {
		Element int
		Until   int
	}
	reqMsg := TableMsg{}
	log.Println("Insider Job Raw MSG: ", message.Body)
	err := json.Unmarshal(message.Body, &reqMsg)
	if err != nil {log.Println("Error Unmarshalling: ", err)}
	log.Println("Insider Job: ", reqMsg)

	number := reqMsg.Element
	until := reqMsg.Until

	for multiplier := 0; multiplier < until; multiplier++ {
		log.Println(number * multiplier)
		time.Sleep(5 * time.Second)
		ch <- []byte(strconv.Itoa(number * multiplier))
	}
	close(ch)
}

func main() {

	// Set up the broker
	broker := workfifo.RabbitMQ{
		User: "guest",
		Password: "guest",
		Host: "localhost",
		Port: 5672,
	}
	// Connect to the broker
	_ = broker.Connect(true)

	// Set up an upstream Queue
	incomingQueue := workfifo.QueueOptions{
		Name:       "leaf",
		Durable:    false,
		AutoDelete: true,
		Exclusive:  false,
		NoWait:     false,
	}

	broker.RPCSubscribe(incomingQueue.Name, PrintTable)
	broker.Disconnect()

}