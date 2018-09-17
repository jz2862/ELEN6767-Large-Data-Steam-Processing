from subprocess import call

file = open("vidNames.txt", "r")
#
#name = []
#for line in file:
#    name = line.split(' \n')[0]
#    call(["youtube-dl", "-o", './movies/%(title)s', name])
call(["youtube-dl", "-o", './movies/%(title)s', 'ZNYmK19-d0U'])
