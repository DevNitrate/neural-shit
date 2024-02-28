use std::fs::File;

use rand::Rng;

const INPUTS: usize = 8;
const HIDDEN: usize = 8;
const OUTPUTS: usize = 2;
const LAYERS: usize = 3;

fn init_weights() -> f64 {
    rand::thread_rng().gen_range(0.0..1.0)
}

struct Network {
    layers: Vec<usize>,
    weights: [Vec<f64>; LAYERS - 1],
    biases: [Vec<f64>; LAYERS - 1],
    data: [Vec<f64>; LAYERS],
    learning_rate: f64
}

impl Network {
    fn new(layers: Vec<usize>, learning_rate: f64, data: [Vec<f64>; LAYERS]) -> Self {
        let mut weights = core::array::from_fn(|_| Vec::new());
        let mut biases = core::array::from_fn(|_| Vec::new());

        for i in 0..layers.len() - 1 {
            for _j in 0..layers[i] {
                weights[i].push(init_weights());
                biases[i].push(0.0);
            }
        }

        Network {
            layers,
            weights,
            biases,
            data,
            learning_rate
        }
    }
}

fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        x * 0.01
    }
}

fn d_relu(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        0.01
    }
}

fn parse_csv(file_path: &str) -> (Vec<[f64; INPUTS]>, Vec<[f64; OUTPUTS]>) {
    let file = File::open(file_path).expect("Could not open file");
    let mut reader = csv::Reader::from_reader(file);

    let mut inputs: Vec<[f64; INPUTS]> = Vec::new();
    let mut outputs: Vec<[f64; OUTPUTS]> = Vec::new();

    let mut input: [f64; INPUTS] = [0.0; INPUTS];
    let mut output: [f64; OUTPUTS] = [0.0; OUTPUTS];
    
    let mut index: u32 = 0;
    for res in reader.records() {
        let record = res.expect("error in parsing");
        
        for f in record.iter() {
            match index {
                0 => {
                    input[index as usize] = f.parse::<f64>().expect("error parsing float");
                    index += 1;
                },
                1 => {
                    input[index as usize] = f.parse::<f64>().expect("error parsing float");
                    index += 1;
                },
                2 => {
                    input[index as usize] = f.parse::<f64>().expect("error parsing float");
                    index += 1;
                },
                3 => {
                    input[index as usize] = f.parse::<f64>().expect("error parsing float");
                    index += 1;
                },
                4 => {
                    input[index as usize] = f.parse::<f64>().expect("error parsing float");
                    index += 1;
                },
                5 => {
                    input[index as usize] = f.parse::<f64>().expect("error parsing float");
                    index += 1;
                },
                6 => {
                    input[index as usize] = f.parse::<f64>().expect("error parsing float");
                    index += 1;
                },
                7 => {
                    index += 1;
                },
                8 => {
                    input[(index as usize) - 1] = f.parse::<f64>().expect("error parsing float");
                    index += 1;
                },
                9 => {
                    output[0] = f.parse::<f64>().expect("error parsing float");
                    index += 1;
                },
                10 => {
                    output[1] = f.parse::<f64>().expect("error parsing float");
                    inputs.push(input);
                    outputs.push(output);
                    index = 0;
                },
                _ => {
                    println!("error");
                }
            }
        }
    }

    
    (inputs, outputs)
}

fn train(inputs: Vec<[f64; INPUTS]>, mut net: Network, outputs: Vec<[f64; OUTPUTS]>) {
    let epochs: u32 = 5;

    for _ in 0..epochs {
        for x in 0..inputs.len() {
            // compute hidden layer
            for i in 0..net.layers[1] {
                let mut activation: f64 = net.biases[0][i];
    
                for k in 0..net.layers[0] {
                    activation += inputs[x][k] * net.weights[0][k];
                }

                net.data[1][i] = relu(activation);
            }

            // compute output layer
            for i in 0..net.layers[2] {
                let mut activation: f64 = net.biases[1][i];
    
                for k in 0..net.layers[1] {
                    activation += inputs[x][k] * net.weights[1][k];
                    net.data[2][i] = relu(activation);
                }
            }

            println!("#############Inputs: {} {} {} {} {} {} {} {} Expected Output: {} {} Predicted: {} {}", inputs[x][0], inputs[x][1], inputs[x][2], inputs[x][3], inputs[x][4], inputs[x][5], inputs[x][6], inputs[x][7], outputs[x][0], outputs[x][1], net.data[2][0], net.data[2][1]);

            // backpropagation

            let mut delta_output: [f64; OUTPUTS] = [0.0; OUTPUTS];

            for i in 0..OUTPUTS {
                let error = outputs[x][i] - net.data[2][i];

                delta_output[i] = error * d_relu(net.data[2][i]);
            }

            let mut delta_hidden: [f64; HIDDEN] = [0.0; HIDDEN];

            for i in 0..HIDDEN {
                let mut error: f64 = 0.0;

                for j in 0..OUTPUTS {
                    error += delta_output[j] * net.weights[1][j];
                }

                delta_hidden[i] = error * d_relu(net.data[1][i]);
            }

            // apply backprop

            for i in 0..OUTPUTS {
                net.biases[1][i] += delta_output[i] * net.learning_rate;

                for j in 0..HIDDEN {
                    net.weights[1][j] += net.data[1][j] * delta_output[i] * net.learning_rate;
                }
            }

            for i in 0..HIDDEN {
                net.biases[0][i] += delta_hidden[i] * net.learning_rate;

                for j in 0..INPUTS {
                    net.weights[0][j] += inputs[x][j] * delta_hidden[i] * net.learning_rate;
                }
            }

            print!("Final hidden weights: ");
            for j in 0..HIDDEN {
                for _k in 0..INPUTS {
                    print!("{}; ", net.weights[0][j]);
                }
            }

            print!("\nFinal output weights: ");
            for j in 0..HIDDEN {
                for _k in 0..INPUTS {
                    print!("{}; ", net.weights[1][j]);
                }
            }

            print!("\nFinal Hidden biases: ");
            for j in 0..HIDDEN {
                print!("{}; ", net.biases[0][j]);
            }

            print!("\nFinal Output biases: ");
            for j in 0..OUTPUTS {
                print!("{}; ", net.biases[1][j]);
            }
            println!()
        }
    }
}

fn main() {
    let inout = parse_csv("test_honey.csv");
    let inputs = inout.0;
    let outputs = inout.1;
    let net: Network = Network::new(vec![INPUTS, HIDDEN, OUTPUTS], 2.0, [vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![0.0, 0.0]]);
    
    train(inputs, net, outputs);
    //println!("{}", - * relu(12.02))
}