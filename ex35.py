class Song(object):
    
    def __init__(self, lyrics):
        self.lyrics = lyrics
    def sing_me_a_song(self):
        for line in self.lyrics:
            print line

happy_bday = Song(["happy birthday to you",
                   "I don't want to get sued",
                   "So I'll stop right there"])

bulls_on_parade = Song(["They rally around the family",
                        "With pockets full of shells"])

my_hearts_will_go_on = Song(["Every night in my dreams",
                             "I see you",
                             "I feel you"])
happy_bday.sing_me_a_song()

bulls_on_parade.sing_me_a_song()

my_hearts_will_go_on.sing_me_a_song()
