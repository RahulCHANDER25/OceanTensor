SRC	=	src/main.cpp 	\
		src/training/fit_training.cpp 	\
		Tensor/MetaData.cpp 	\
		Network/Linear/LinearNetwork.cpp 	\
		Network/Sequential.cpp

OBJ	=	$(SRC:.cpp=.o)

CC	:=	g++
RM	:=	rm -rf
CPPFLAGS	=	-iquote Tensor/ -iquote Network/ -iquote src/include -std=c++20 -g3
CFLAGS	=	-Wall

NAME	=	oceanTensor

all: $(NAME)

$(NAME):	$(OBJ)
	$(CC) -o $(NAME) $(OBJ) $(CFLAGS)

clean:
	$(RM) $(OBJ)

fclean: clean
	$(RM) $(NAME)

re: fclean all

.PHONY: fclean re all clean
