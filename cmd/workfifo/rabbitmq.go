package workfifo

import (
	"github.com/streadway/amqp"
	"log"
	"strconv"
)

type ExchangeOptions struct {
	Name string
	Type string
	Durable bool
	AutoDelete bool
	Internal bool
	NoWait bool
}

type QueueOptions struct {
	Name string
	Durable bool
	AutoDelete bool
	Exclusive bool
	NoWait bool
}

type RabbitMQ struct {
	connection *amqp.Connection
	Channel    *amqp.Channel
	User string
	Password string
	Port int
	Host string
	address    string
}

func (rabbit *RabbitMQ) Connect(openChannel bool) (err error) {
	rabbit.address = concatenateStrings("amqp://",
		rabbit.User, ":", rabbit.Password, "@", rabbit.Host, ":", strconv.Itoa(rabbit.Port), "/")
	if rabbit.connection, err = amqp.Dial(rabbit.address); err != nil {
		log.Printf("Error Occured while Connecting: %v\n", err)
	}

	if openChannel {
		err = rabbit.OpenChannel()
	}

	return
}

func (rabbit *RabbitMQ) OpenChannel() (err error) {
	if channel, err := rabbit.connection.Channel(); err != nil {
		log.Printf("Error Occured while Opening Channel: %v\n", err)
		return err
	} else {
		rabbit.Channel = channel
	}
	return
}

func (rabbit *RabbitMQ) CloseChannel() (err error) {
	if err = rabbit.Channel.Close(); err != nil {
		log.Printf("Error Occured while closing Channel: %v\n", err)
	}
	return
}

func (rabbit *RabbitMQ) Disconnect() (err error) {

	rabbit.CloseChannel()
	if err = rabbit.connection.Close(); err != nil {
		log.Printf("Error Occured while Disconnecting: %v\n", err)
	}
	return

}

func (rabbit *RabbitMQ) SetupExchange(exchangeOptions ExchangeOptions) (err error) {

	// Declare an Exchange
	err = rabbit.Channel.ExchangeDeclare(
		exchangeOptions.Name,
		exchangeOptions.Type,
		exchangeOptions.Durable,
		exchangeOptions.AutoDelete,
		exchangeOptions.Internal,
		exchangeOptions.NoWait,
		nil,
	)
	if err != nil { log.Printf("Error Occured while declaring Exchange %s: %v\n", exchangeOptions.Name, err) }

	return
}

func (rabbit *RabbitMQ) SetupQueue(options QueueOptions) (queue amqp.Queue, err error) {
	queue, err = rabbit.Channel.QueueDeclare(options.Name,
		options.Durable,
		options.AutoDelete,
		options.Exclusive, options.NoWait, nil)

	return
}


func (rabbit *RabbitMQ) RPCPublish(callbackQueueName string,
	exchangeName string, routingKey string,
	message Message) (msgsChannel <-chan amqp.Delivery, correlationId string, err error) {

	// Publish to the Queue
	correlationId = randomString(32)
	err = rabbit.Channel.Publish(
		exchangeName,
		routingKey, // routing key
		false,
		false,
		amqp.Publishing{
			ContentType:   message.ContentType,
			CorrelationId: correlationId,
			ReplyTo:       callbackQueueName,
			Body:          message.Body,
			},
		)

	log.Printf("Message sent to exchange %s\n", exchangeName)

	// Create a Consumer
	msgsChannel, err = rabbit.Channel.Consume(
		callbackQueueName, // callbackQueue
		"",                 // consumer
		true,               // auto-ack
		false,              // exclusive
		false,              // no-local
		false,              // no-wait
		nil,                // args
	)

	if err != nil {
		log.Printf("Error Occured while declaring a RPC Client Consumer: %v\n", err)
	}

	return
}

func (rabbit *RabbitMQ) RPCSubscribe(queueName string, handler Handler) (err error) {

	ch, _ := rabbit.connection.Channel()

	// Set Channel Params
	if err = ch.Qos(
		1,     // prefetch count
		0,     // prefetch size
		false, // global
	); err != nil {
		log.Printf("Error Occured while setting QoS: %v\n", err)
	}

	// Get the messages Channel
	msgs, err := ch.Consume(
		queueName, // queue
		"",     // consumer
		false,  // auto-ack
		false,  // exclusive
		false,  // no-local
		false,  // no-wait
		nil,    // args
	)
	if err != nil {
		log.Printf("Failed to register a consumer: %v\n", err)
	}

	// Get one message from the Channel
	msg := <-msgs

	// Create an empty channel of bytes
	responseChannel := make(chan []byte)
	go handler(Message{
		Headers: nil,
		Body: msg.Body,
	}, responseChannel)

	if err != nil { log.Printf("Failed to fetch the message: %v\n", err) }

	for response := range responseChannel{
		log.Println("Sending: ", response, " to: ", msg.ReplyTo)
		// Publish Response back
		if err = ch.Publish(
			"",
			msg.ReplyTo,
			false,
			false,
			amqp.Publishing{
				ContentType: "text/plain",
				CorrelationId: msg.CorrelationId,
				Body: response,
			},
		); err != nil {
			log.Printf("Failed to publish the response back: %v\n", err)
		}
		//<- responseChannel
	}

	// Ack the message
	if err = msg.Ack(false); err != nil {
		log.Printf("Error Acking Message: %v\n", err)
	}

	return err

}




