
gates = ["Controlled-X", "Controlled-V", "Controlled-V+"]

circuits = {
    "MKG": [
        ("Controlled-X", (0, 3)),
        ("Controlled-V", (1, 0)),
        ("Controlled-V", (1, 3)),
        ("Controlled-X", (0, 3)),
        ("Controlled-V+", (1, 0)),
        ("Controlled-X", (2, 1)),
        ("Controlled-V", (1, 2)),
        ("Controlled-V", (1, 2)),
        ("Controlled-X", (2, 1)),
        ("Controlled-V", (3, 2)),
        ("Controlled-V", (3, 1)),
        ("Controlled-X", (2, 1)),
        ("Controlled-V+", (3, 2)),
        ("Controlled-V", (3, 2)),
        ("Controlled-V", (3, 0)),
        ("Controlled-X", (2, 0)),
        ("Controlled-V+", (3, 2)),
    ],
}