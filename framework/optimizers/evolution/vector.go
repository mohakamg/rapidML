package evolution

type VectorRepresentation struct {
	representation
	parents []*VectorRepresentation
	Candidate []float32
}
