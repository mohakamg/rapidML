package main

import (
	"../../cmd/workfifo"
	"encoding/json"
	"log"
)

type TableMsg struct {
	Element int
	Until   int
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
	
	// Set up an exchange
	exchange := workfifo.ExchangeOptions{
		Name:       "runnerjobs",
		Type:       "direct",
		Durable:    false,
		AutoDelete: false,
		Internal:   false,
		NoWait:     false,
	}
	
	// Set up an upstream Queue 1
	upstreamQueue1 := workfifo.QueueOptions{
		Name:       "cronous",
		Durable:    false,
		AutoDelete: true,
		Exclusive:  false,
		NoWait:     false,
	}
	// Set up a downstream Queue 1
	downstreamQueue1 := workfifo.QueueOptions{
		Name:       "downstream_cronous",
		Durable:    false,
		AutoDelete: true,
		Exclusive:  false,
		NoWait:     false,
	}

	// Set up an upstream Queue 1
	upstreamQueue2 := workfifo.QueueOptions{
		Name:       "leaf",
		Durable:    false,
		AutoDelete: true,
		Exclusive:  false,
		NoWait:     false,
	}
	// Set up a downstream Queue 1
	downstreamQueue2 := workfifo.QueueOptions{
		Name:       "downstream_leaf",
		Durable:    false,
		AutoDelete: true,
		Exclusive:  false,
		NoWait:     false,
	}

	// Declare Exchange and Queues
	broker.SetupExchange(exchange)
	broker.SetupQueue(upstreamQueue1)
	broker.SetupQueue(downstreamQueue1)
	broker.SetupQueue(upstreamQueue2)
	broker.SetupQueue(downstreamQueue2)

	// Set Up Binding
	crounous_binding_key := "cronous_jobs"
	broker.Channel.QueueBind(
		upstreamQueue1.Name,
		crounous_binding_key,
		exchange.Name, false, nil,
	)
	leaf_binding_key := "leaf_jobs"
	broker.Channel.QueueBind(
		upstreamQueue2.Name,
		leaf_binding_key,
		exchange.Name, false, nil,
	)


	msg := TableMsg{
		Element: 7,
		Until:   10,
	}
	msgBody, _ := json.Marshal(msg)
	log.Println("Msg: ", msgBody)
	responseChannel, publishedMessageCorrelationID, _ := broker.RPCPublish(downstreamQueue2.Name, exchange.Name, leaf_binding_key, workfifo.Message{
		ContentType: "text/plain",
		Headers:     nil,
		Body:        msgBody,
	})

	for response := range responseChannel {
		if response.CorrelationId == publishedMessageCorrelationID {
			log.Println(string(response.Body))
		}
	}

	broker.Disconnect()

}
