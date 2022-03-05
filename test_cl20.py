gates = ["Controlled-X", "Controlled-V", "Controlled-V+"]

circuits = {
    "TSG":[
        ("Controlled-V", (2, 1)),
        ("Controlled-V", (2, 0)),
        ("Controlled-X", (1, 0)),
        ("Controlled-V+", (2, 1)),
        ("Controlled-X", (1, 0)),
        ("Controlled-V", (1, 0)),
        ("Controlled-V", (1, 2)),
        ("Controlled-X", (0, 2)),
        ("Controlled-V+", (1, 0)),
        ("Controlled-X", (0, 2)),
        ("Controlled-X", (1, 2)),
        ("Controlled-X", (1, 0)),
        ("Controlled-V", (2, 3)),
        ("Controlled-V", (2, 1)),
        ("Controlled-X", (3, 1)),
        ("Controlled-V+", (2, 3)),
        ("Controlled-X", (3, 2)),
        ("Controlled-V", (2, 3)),
        ("Controlled-V", (2, 3)),
        ("Controlled-X", (2, 3)),
    ]
}
"""
circuits = {
    "TSG": [
        ("Controlled-V", (1, 2)),
        ("Controlled-X", (2, 0)),
        ("Controlled-V+", (1, 0)),
        ("Controlled-V", (1, 2)),
        ("Controlled-V+", (2, 1)),
        ("Controlled-V", (2, 0)),
        ("Controlled-X", (1, 0)),
        ("Controlled-V+", (2, 3)),
        ("Controlled-X", (3, 1)),
        ("Controlled-V", (2, 3)),
        ("Controlled-X", (2, 3)),
        ("Controlled-X", (3, 2)),
        ("Controlled-X", (2, 3)),
    ]
}
"""