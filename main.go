package main

import (
	"encoding/gob"
	"flag"
	"log"
	"os"
	"time"

	"github.com/cheggaaa/pb"
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func fizzbuzz(n int) []float64 {
	if n%15 == 0 {
		return []float64{0, 0, 0, 1}
	}
	if n%3 == 0 {
		return []float64{0, 0, 1, 0}
	}
	if n%5 == 0 {
		return []float64{0, 1, 0, 0}
	}
	return []float64{1, 0, 0, 0}
}

func bin(n int) []float64 {
	var r [10]float64
	for d := 0; d < 10; d++ {
		r[d] = float64(n >> d & 1)
	}
	return r[:]
}

func RMS(yHat, y gorgonia.Input) (retVal *gorgonia.Node, err error) {
	if err = gorgonia.CheckOne(yHat); err != nil {
		return nil, errors.Wrap(err, "unable to extract node from yHat")
	}

	if err = gorgonia.CheckOne(y); err != nil {
		return nil, errors.Wrap(err, "unable to extract node from y")
	}

	if retVal, err = gorgonia.Sub(yHat.Node(), y.Node()); err != nil {
		return nil, errors.Wrap(err, "(ŷ-y)")
	}

	if retVal, err = gorgonia.Square(retVal); err != nil {
		return nil, errors.Wrap(err, "(ŷ-y)²")
	}

	if retVal, err = gorgonia.Mean(retVal); err != nil {
		return nil, errors.Wrap(err, "mean((ŷ-y)²)")
	}

	return
}

func main() {
	epochs := flag.Int("epochs", 2000, "number of epoch")
	batchsize := flag.Int("batchsize", 120, "size of batch")
	flag.Parse()

	bs := *batchsize
	g := gorgonia.NewGraph()
	m := newNN(g)

	x := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithName("X"),
		gorgonia.WithShape(bs, 10),
	)
	y := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithName("y"),
		gorgonia.WithShape(bs, 4),
	)

	if err := m.fwd(x); err != nil {
		log.Fatalf("%+v", err)
	}

	cost := gorgonia.Must(RMS(m.out, y))

	if _, err := gorgonia.Grad(cost, m.learnables()...); err != nil {
		log.Fatal(err)
	}

	vm := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(m.learnables()...))
	solver := gorgonia.NewAdamSolver(gorgonia.WithLearnRate(0.001))

	numExamples := 1024
	batches := numExamples / bs

	bar := pb.New(*epochs)
	bar.SetRefreshRate(time.Second / 20)
	bar.Set("Training", 0)
	bar.Start()

	for i := 0; i < *epochs; i++ {
		for b := 0; b < batches; b++ {
			start := b * bs
			end := start + bs
			if start >= numExamples {
				break
			}
			if end > numExamples {
				end = numExamples
			}

			xB := make([]float64, (end-start)*10)
			yB := make([]float64, (end-start)*4)
			for n := 0; n < end-start; n++ {
				copy(xB[n*10:], bin(n+31+start))
				copy(yB[n*4:], fizzbuzz(n+31+start))
			}

			xT := tensor.New(tensor.WithShape(bs, 10), tensor.WithBacking(xB))
			gorgonia.Let(x, xT)

			yT := tensor.New(tensor.WithShape(bs, 4), tensor.WithBacking(yB))
			gorgonia.Let(y, yT)

			if err := vm.RunAll(); err != nil {
				log.Fatalf("Failed at inter  %d: %v", b, err)
			}
			solver.Step(gorgonia.NodesToValueGrads(m.learnables()))
			vm.Reset()
		}
		bar.Increment()
	}
	bar.Finish()

	err := save(m.learnables())
	if err != nil {
		log.Fatal(err)
	}
}

func save(nodes []*gorgonia.Node) error {
	f, err := os.Create("fizzbuzz.bin")
	if err != nil {
		return err
	}
	defer f.Close()
	enc := gob.NewEncoder(f)
	for _, node := range nodes {
		err := enc.Encode(node.Value())
		if err != nil {
			return err
		}
	}
	return nil
}
