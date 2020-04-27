package main

import (
	"encoding/gob"
	"fmt"
	"log"
	"os"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func bin(n int) []float64 {
	var r [10]float64
	for d := 0; d < 10; d++ {
		r[d] = float64(n >> d & 1)
	}
	return r[:]
}

func dec(v []float64, n int) string {
	m := v[0]
	j := 0
	for i, vv := range v {
		if m < vv {
			j = i
			m = vv
		}
	}
	switch j {
	case 1:
		return "Buzz"
	case 2:
		return "Fizz"
	case 3:
		return "FizzBuzz"
	}
	return fmt.Sprint(n)
}

func main() {
	g := gorgonia.NewGraph()
	values := make([]float64, 10)
	xT := tensor.New(tensor.WithBacking(values))
	x := gorgonia.NodeFromAny(g, xT, gorgonia.WithName("x"))

	var wT0, wT1 *tensor.Dense
	f, err := os.Open(`../../fizzbuzz.bin`)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	decoder := gob.NewDecoder(f)
	err = decoder.Decode(&wT0)
	if err != nil {
		log.Fatal(err)
	}
	err = decoder.Decode(&wT1)
	if err != nil {
		log.Fatal(err)
	}
	w0 := gorgonia.NodeFromAny(g, wT0, gorgonia.WithName("w0"))
	w1 := gorgonia.NodeFromAny(g, wT1, gorgonia.WithName("w1"))

	l0dot := gorgonia.Must(gorgonia.Mul(x, w0))
	l1 := gorgonia.Must(gorgonia.Rectify(l0dot))
	l1dot := gorgonia.Must(gorgonia.Mul(l1, w1))
	y := gorgonia.Must(gorgonia.Sigmoid(l1dot))

	vm := gorgonia.NewTapeMachine(g)

	vm.Reset()
	for i := 1; i <= 100; i++ {
		copy(values, bin(i))
		if err = vm.RunAll(); err != nil {
			log.Fatal(err)
		}
		vm.Reset()
		fmt.Println(dec(y.Value().Data().([]float64), i))
	}
}
