# -*- coding: utf-8 -*-
import nose
import yutils.text as text


def string_equal(a, b):
    print 'a =', a
    print 'b =', b
    assert a == b

def test_titles():
    # titles stolen from http://www.harvard.com/onourshelves/top100.html
    test_strings = ["A People's History of the United States",
                    "Howard Zinn",
                    "The Wind Up Bird Chronicles",
                    "Haruki Murakami",
                    "The New York Trilogy",
                    "Paul Auster",
                    "The Crying of Lot 49",
                    "Thomas Pynchon",
                    "Lord of the Rings",
                    "J.R.R. Tolkien",
                    "Jane Eyre",
                    "Charlotte Bronte",
                    "Lolita",
                    "Vladimir Nabokov",
                    "Nineteen Eighty-Four",
                    "George Orwell",
                    "One Hundred Years of Solitude",
                    "Gabriel Garcia Marquez",
                    "The Catcher in the Rye",
                    "J.D. Salinger",
                    "Crime and Punishment Dostoevsky",
                    "On the Road Kerouac",
                    "Alice in Wonderland Carrol",
                    "Brothers Karamozov Dostoevsky",
                    "The Age of Innocence Wharton",
                    "Don Quixote Cervantes",
                    "Perfume Suskind",
                    "Ulysses Joyce",
                    "Anna Karenina Tolstoy",
                    "Complete Stories of Flannery O'Connor",
                    "Cry the Beloved Country Paton",
                    "Dracula Stoker",
                    "The Eagles Die Marek",
                    "Emotionally Weird Atkinson",
                    "The Handmaid's Tale Atwood",
                    "Infinite Jest Wallace",
                    "Kitchen Yoshimoto",
                    "London Fields Amis",
                    "Moise and the World of Reason Williams",
                    "Movie Wars Rosenbaum",
                    "Paradise Lost Milton",
                    "Persuasion Austen",
                    "Tortilla Curtain Boyle",
                    "Visions of Excess Bataille",
                    "Where the Wild Things Are Sendak",
                    "Wild Sheep Chase Murakami",
                    "Beloved Morrison",
                    "Counterfeiters Gide",
                    "The Bell Jar Plath",
                    "Blind Owl Hedayat",
                    "Complete Works of Edgar Allen Poe",
                    "The Count of Monte Cristo Dumas",
                    "Dealing with Dragons Wrede",
                    "The Earthsea Trilogy Le Guin",
                    "The Ecology of Fear Davis",
                    "Franny and Zooey Salinger",
                    "History of the Peloponnesian War Thucydides",
                    "How the Garcia Girls Lost their Accents Alvarez",
                    "Kabuki: Circle of Blood Mack & Jiang",
                    "Of Human Bondage Maugham"]

    for s in test_strings:
        a = s.lower()
        yield string_equal, text.titlecase(a), s


if __name__ == '__main__':
    nose.runmodule()
