include t2t_docsys/Makefile.include

.PHONY: all doc install
all: doc
doc: readme.txt readme.md manual.txt manual.md manual.html
install: doc
	cp readme.txt ../README
	cp readme.md ../README.md
	cp manual.txt ../MANUAL
	cp manual.md ../MANUAL.md
	cp manual.html ../MANUAL.html

%.html: %.t2t $(wildcard *.t2t)
	$(call t2t_docsys_html,$@,$<)

%.txt: %.t2t $(wildcard *.t2t)
	$(call t2t_docsys_txt,$@,$<)

%.md: %.t2t $(wildcard *.t2t)
	$(call t2t_docsys_md,$@,$<)

.PHONY: clean
clean:
	@rm -f $(T2T_DOCSYS_OUTPUT_WILDCARDS)
