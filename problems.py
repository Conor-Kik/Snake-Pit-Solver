from dataclasses import dataclass


@dataclass
class ProblemData:
    T: range
    grid: list
    circle_squares: list
    x_squares: list


PROBLEMS = {

    1: ProblemData(
        T=range(2, 7),
        grid=[
            [None, None, 2,   None, 2,   4],
            [2,    None, 2,   4,    None, None],
            [None, None, None, None, None, None],
            [None, None, None, None, None, None],
            [None, None, None, 2,    4,    None],
            [None, 2,    None, None, None, None],
        ],
        circle_squares=[
            (0, 3),
            (1, 1),
            (5, 2),
        ],
        x_squares=[],
    ),

    2: ProblemData(
        T=range(2, 14),
        grid=[[None]*6 for _ in range(6)],
        circle_squares=[
            (0, 0), (0, 1), (0, 3), (0, 4), (0, 5),
            (1, 0), (1, 1), (1, 3), (1, 4), (1, 5),
            (2, 0), (2, 1),
        ],
        x_squares=[
            (3, 4), (3, 5),
            (4, 0), (4, 1), (4, 2), (4, 4), (4, 5),
            (5, 0), (5, 1), (5, 2), (5, 4), (5, 5),
        ],
    ),

    3: ProblemData(
        T=range(2,9),
        grid=[
            [2, None,None, 6, None, None, None, 6,None, None, 6],
            [None, None, None, None, None, 6, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None, None, None],
            [6, None,None, 6, None, 6, None, 4,None, None, 2],
            [None, None, None, None, 4, None, 4, None, None, None, None],
            [None, 6, None, None, None, None, None, None, None, 5, None],
            [None, None, None, None, 3, None, 2, None, None, None, None],
            [6, None,None, 4, None, 3, None, 3,None, None, 3],
            [None, None, None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, 5, None, None, None, None, None],
            [5, None, None, 5, None, None, None, 2, None, None, 3],
        ],
        circle_squares=[
            (0, 0), (0, 3), (0, 7), (0, 10),
            (3, 0), (3, 3), (3, 7), (3, 10),
            (7, 0), (7, 3), (7, 7), (7, 10),
            (10, 0), (10, 3), (10, 7), (10, 10),
            (5, 2), (5, 8),
        ],
        x_squares=[
            (1,1), (1,2),(1,8),(1,9),
            (2,1), (2,2),(2,8),(2,9),(2,5),
            (4,5),(5,4),(5,6),(6,5),
            (8,1), (8,2),(8,8),(8,9),(8,5),
            (9,1), (9,2),(9,8),(9,9),
        ],
    ),

    4: ProblemData(
        T=range(2, 10),
        grid=[
            [None, 3, None, None, None, None, None, None, None, None],
            [None, None, 4, None, None, None, None, None, None, 4],
            [None, None, None, 5, None, None, None, None, 5, None],
            [None, None, None, None, 6, None, None, 6, None, None],
            [None, None, None, None, None, None, None, None, 7, None],
            [None, 5, None, None, 9, None, None, None, None, None],
            [None, None, 4, None, None, 8, None, None, None, None],
            [None, 3, None, None, None, None, 7, None, None, None],
            [2, None, None, None, None, None, None, 6, None, None],
            [None, None, None, None, None, None, None, None, 5, None],
        ],
        circle_squares=[
            (0, 2), (0, 6), (1, 8),
            (2, 0), (2, 2), (2, 4),
            (3, 6), (3, 9),
            (4, 2), (5, 7),
            (6, 0), (6, 3),
            (7, 5), (7, 9),
            (8, 1),
            (9, 3), (9, 7),
        ],
        x_squares=[
            (0, 3), (0, 7),
            (1, 1), (1, 7),
            (2, 5), (2, 9),
            (3, 0),
            (6, 9),
            (7, 0), (7, 4),
            (8, 2), (8, 8),
            (9, 2), (9, 6),
        ],
    ),

    5: ProblemData(
        T=range(2, 17),
        grid=[
            [None, None, None, 16, None, 8, None, 16],
            [None]*8,
            [None, None, None, 7, None, 8, None, 2],
            [None]*8,
            [None]*8,
            [7, None, 16, None, 2, None, None, None],
            [None]*8,
            [16, None, 16, None, 8, None, None, None],
        ],
        circle_squares=[
            (0, 5), (0, 7),
            (2, 7),
            (5, 0), (5, 2), (5, 4),
            (7, 4),
        ],
        x_squares=[],
    ),

    6: ProblemData(
        T=range(2, 10),
        grid=[[None]*12 for _ in range(12)],
        circle_squares=[
            (0, 1),(0, 3),(0, 5),(0, 7),(0, 9),(0, 11),
            (1, 1),(1, 3),(1, 5),(1, 9),(1, 11),
            (2, 0),(2, 3),(2, 4),(2, 8),(2, 11),
            (3, 3),(3, 5),(3, 6),(3, 8),(3, 11),
            (4, 3),(4, 5),(4, 6),(4, 8),
            (5, 0),(5, 1),(5, 3),(5, 4),(5, 6),(5, 9),(5, 10),
            (6, 0),(6, 3),(6, 4),(6, 6),(6, 9),
            (7, 0),(7, 1),(7, 2),(7, 4),(7, 7),(7, 9),(7, 10),(7, 11),
            (8, 1),(8, 2),(8, 4),(8, 7),(8, 9),(8, 10),(8, 11),
            (9, 1),(9, 2),(9, 4),(9, 5),(9, 6),(9, 9),(9, 10),
            (10, 1),(10, 3),(10, 4),(10, 5),(10, 7),(10, 9),(10, 11),
            (11, 1),(11, 3),(11, 5),(11, 7),(11, 9),(11, 11),
        ],
        x_squares=[
            (0,0),(0,2),(0,4),(0,6),(0,8),(0,10),
            (1,0),(1,2),(1,4),(1,6),(1,7),(1,8),(1,10),
            (2,1),(2,2),(2,5),(2,6),(2,7),(2,9),(2,10),
            (3,0),(3,1),(3,2),(3,4),(3,7),(3,9),(3,10),
            (4,0),(4,1),(4,2),(4,4),(4,7),(4,9),(4,10),(4,11),
            (5,2),(5,5),(5,7),(5,8),(5,11),
            (6,1),(6,2),(6,5),(6,7),(6,8),(6,10),(6,11),
            (7,3),(7,5),(7,6),(7,8),
            (8,0),(8,3),(8,5),(8,6),(8,8),
            (9,0),(9,3),(9,7),(9,8),(9,11),
            (10,0),(10,2),(10,6),(10,8),(10,10),
            (11,0),(11,2),(11,4),(11,6),(11,8),(11,10),
        ],
    ),
    7: ProblemData(
        T=range(2, 21),
        grid=[
            [None, None, None, None, None, None, None, None, None, None], 
            [None, None, 20,   5,    None, 3,    8,    None, None, None],  
            [None, None, 3,    None, None, None, None, None, None, None],  
            [None, None, 5,    None, None, None, None, None, None, None],  
            [None, None, 4,    None, None, None, None, None, None, None],  
            [None, None, 2,    4,    8,    None, 3,    5,    None, None],  
            [None, None, 4,    None, None, None, None, None, None, None],  
            [None, None, 2,    None, None, None, None, None, None, None],  
            [None, None, 3,    None, None, None, None, None, None, None],  
            [None, None, 6,    3,    5,    None, 5,    3,    None, None],  
        ],

        circle_squares=[
            (1, 0),   
            (1, 2),   
            (1, 4),  
            (1, 7),  
            (1, 9),  

            (5, 4),  
            (5, 5),  
            (5, 9),   

            (7, 9),  

            (9, 0),   
            (9, 5),   
            (9, 6), 
            (9, 9),   
        ],

        x_squares=[]
    ),
    10: ProblemData(
        T=range(2, 20),
        grid = [
            [None, None, None, None, None, None, None, None, None, None, None, None], 
            [None, None, None, None, None, 3,    15,   None, None, None, None, None], 
            [None, 2,    None, None, 3,    None, None, 13,   None, None, None, None],  
            [None, None, None, 7,    None, None, None, None, 3,    None, None, None],  
            [None, None, 5,    None, None, None, None, None, None, 3,    None, None],  
            [None, 7,    None, None, None, None, None, None, None, None, 14,   None],  
            [5,    7,    3,    13,   2,    5,    5,    None, 3,    3,    5,    None], 
            [None, None, None, None, None, None, None, None, None, None, None, None], 
            [3,    None, None, None, None, None, None, None, None, None, None, 3],     
            [3,    None, None, None, None, None, None, None, None, None, None, 6],    
            [19,   None, None, None, 5,    5,    None, 3,    None, None, None, 3],    
            [3,    None, None, None, None, None, None, None, None, None, None, None], 
        ],


        circle_squares = [
            (1, 6),     
            (2, 10),    
            (5, 10),     
            (6, 6),     
            (6, 7),      
            (6, 11),     
            (7, 0),     
            (7, 11),    
            (10, 0),    
            (10, 6),    
            (11, 11),   
        ],

        x_squares = []
    ),
    9 : ProblemData(

        T=range(2, 10),
        grid=[[None]*12 for _ in range(12)],
        circle_squares=[],
        x_squares=[]
        
    ),
    9: ProblemData(
        T=range(2, 9),
        grid=[
            [None, None, None, None,  8, None, None, None, None, None],  
            [  8, None, None, None, None, None, None, None, None, None], 
            [None, None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None,  8, None, None, None],  
            [None, None, None, None,  8, None, None, None, None,  8],    
            [None, None,  8, None, None, None, None, None, None, None],  
            [None,  8, None, None, None, None, None, None, None, None], 
            [None, None,  8, None, None, None, None, None, None, None],  
        ],
        circle_squares=[
            (0, 4),
            (1, 0),
            (5, 6),
            (6, 4),
            (6, 9),
            (7, 2),
            (8, 1),
            (9, 2),

            (1, 5),
            (2, 4),
            (3, 0),
            (4, 7),
            (6, 7),
            (8, 6),
            (8, 9),
            (9, 5),
        ],
        x_squares=[],
    )
}


def get_problem(problem_id: int) -> ProblemData:
    if problem_id not in PROBLEMS:
        raise ValueError(f"Unknown problem {problem_id}")
    return PROBLEMS[problem_id]
