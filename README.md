# kdl-script

Haven't you always wanted to execute [a KDL file](https://kdl.dev/)? Well now you can!


# Demo

```
cargo run examples/simple.kdl

{
  y: 22
  x: 11
}
33
```

Which is executing the following kdl document:


```kdl
@derive "Display"
struct "Point" {
    x "f64"
    y "f64"
}

fn "main" {
    outputs { _ "f64"; }

    let "pt1" "Point" {
        x 1.0
        y 2.0
    }
    let "pt2" "Point" {
        x 10.0
        y 20.0
    }
    
    let "sum" "add:" "pt1" "pt2"
    print "sum"

    return "+:" "sum.x" "sum.y"
}

fn "add" {
    inputs { a "Point"; b "Point"; }
    outputs { _ "Point"; }

    return "Point" {
        x "+:" "a.x" "b.x"
        y "+:" "a.y" "b.y"
    }
}
```


# Why

To spite parsers