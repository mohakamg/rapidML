package main

import (
	"fmt"
	"reflect"
)

type Base struct {
	Xxx int
}

type Inherited struct {
	Base
	Yyy int
}

func returnInherited() interface{} {
	return Inherited{
		Base: Base{Xxx: 1},
		Yyy:  0,
	}
}


func main() {

	base := Base{Xxx: 1}
	inherited := returnInherited()
	fmt.Println(reflect.TypeOf(base))
	fmt.Println(reflect.TypeOf(inherited))

	fmt.Println(reflect.TypeOf(base) == reflect.TypeOf(inherited))
	
}
